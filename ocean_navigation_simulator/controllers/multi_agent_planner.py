import concurrent.futures
import dataclasses
import multiprocessing as mp
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr
import networkx as nx

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJPlannerBase,
    HJReach2DPlanner,
)
from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import (
    PlatformAction,
    PlatformActionSet,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    PlatformStateSet,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.environment.MultiAgent import MultiAgent
import scipy.integrate as integrate

class DecentralizedReactiveControl:
    def __init__(
        self,
        observation: ArenaObservation,
        param_dict: dict,
        platform_dict: dict,
        nb_max_neighbors: int = 2,
    ):
        self.nb_max_neighbors = nb_max_neighbors
        self.param_dict = param_dict
        # perform computation in m:
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(unit="m", graph_type="complete")
        # argspartition ensures that the all elements before nb_max_neighbors are the smallest elements
        self.g_a, self.g_b = None, None
        self.observation = observation
        self.u_max_mps = platform_dict["u_max_in_mps"]

    def get_reactive_control(self, pltf_id, hj_optimal_action):
        self._set_constraint_g(pltf_id=pltf_id)
        a, b = self._set_attraction_or_repulsion(self.g_a), self._set_attraction_or_repulsion(
            self.g_b
        )
        if a == 0 and b == 0:  # GoToGoal
            u_i = hj_optimal_action
        elif (
            self.g_a > -self.param_dict["delta_1"] ** 2
            or self.g_b > -self.param_dict["delta_1"] ** 2
        ):  # achieve connectivity
            u_i = self._compute_potential_force(pltf_id=pltf_id, a=a, b=b)
        else:  # maintain connectivity
            u_i = self._compute_potential_force(
                pltf_id=pltf_id, a=a, b=b
            ) + hj_optimal_action.scaling(self.param_dict["k_2"])
        return u_i

    def _compute_potential_force(self, pltf_id, a, b):
        potential_force = -self.param_dict["k_1"] * (
            a * self._compute_gradient_g(pltf_id, self.pltf_a)
            + b * self._compute_gradient_g(pltf_id, self.pltf_b)
        )
        return PlatformAction(
            magnitude=np.linalg.norm(potential_force, ord=2)
            / self.u_max_mps,  # scale in % of max u
            direction=np.arctan2(potential_force[1], potential_force[0]),
        )

    def _compute_gradient_g(self, pltf_id, d_x_id):
        # units of hj is m/s and rad/s
        return 2 * np.array(
            [
                self.observation[pltf_id].platform_state.lon.m
                - self.observation[d_x_id].platform_state.lon.m,
                self.observation[pltf_id].platform_state.lat.m
                - self.observation[d_x_id].platform_state.lat.m,
            ]
        )

    def _set_constraint_g(self, pltf_id):
        # obtained ordered list of neighbors (ascendent by distance) until element self.nb_max_neighbors
        # faster than a sort over whole array
        ordered_dist_neighbors = np.argpartition(
            self.adjacency_mat[pltf_id, :], self.nb_max_neighbors
        )
        # Extract closest platforms id: start at idx=1 since diagonal elements have distance 0 and correspond to self-loops
        self.pltf_a, self.pltf_b = ordered_dist_neighbors[1 : self.nb_max_neighbors + 1]
        self.g_a = (
            self.adjacency_mat[pltf_id, self.pltf_a] ** 2
            - self.param_dict["communication_thrsld"] ** 2
        )
        self.g_b = (
            self.adjacency_mat[pltf_id, self.pltf_b] ** 2
            - self.param_dict["communication_thrsld"] ** 2
        )

    def _set_attraction_or_repulsion(self, g):
        if g <= -self.param_dict["delta_3"] ** 2:
            return -1
        elif g >= -self.param_dict["delta_2"] ** 2:
            return 1
        else:  # delta_3 < g < delta_2
            return 0


class FlockingControl:
    def __init__(
        self,
        observation: ArenaObservation,
        param_dict: dict,
        platform_dict: dict,
    ):
        self.param_dict = param_dict
        self.G_proximity = observation.graph_obs.G_communication
        self.observation = observation
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(unit="m", graph_type="communication")# get adjacency
        self.u_max_mps = platform_dict["u_max_in_mps"]
        self.dt_in_s = platform_dict["dt_in_s"]
        self.r_alpha = self.sigma_norm(self.param_dict["interaction_range"])
        self.d_alpha = self.sigma_norm(self.param_dict["ideal_distance"])

    def sigma_norm(self, z_norm):
        val = self.param_dict["epsilon"]*(z_norm**2)
        return 1/self.param_dict["epsilon"] * (np.sqrt(1 + val)-1)

    def sigma_1(self,z):
        return z/np.sqrt(1+z**2)

    def bump_function(self, z):
        #maybe can vectorize it directly with np.where(condition, val if true, val if false) but we have 3 different vals to assign for 
        #3 different conditions::
        # z = np.where(np.logical_and(z>=0, z < self.param_dict["h"]), 1, z)
        if 0 <= z < self.param_dict["h"]:
            return 1
        elif self.param_dict["h"] <= z < 1:
            arg = np.pi*(z-self.param_dict["h"])/(1-self.param_dict["h"])
            return 0.5*(1+np.cos(arg))
        else:
            return 0
        
    def plot_psi_and_phi_alpha(self, step:Optional[int] = 100, savefig: Optional[bool]=False):
        z_range = np.arange(start= 0, stop= self.r_alpha, step=step)
        phi_alpha = [self.phi_alpha(z=z) for z in z_range]
        psi_alpha = [integrate.quad(self.phi_alpha, self.d_alpha, z)[0] for z in z_range]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(z_range, phi_alpha)
        ax1.set_ylabel(r'$\phi_{\alpha}$')
        ax1.set_xticks([self.d_alpha, self.r_alpha])
        ax1.set_xticklabels([r'$d_{\alpha}$', r'$r_{\alpha}$'])
        ax2.plot(z_range, psi_alpha)
        ax2.set_ylabel(r'$\psi_{\alpha}$')
        ax2.set_xlabel(r'$\Vert z \Vert_{\sigma}$')
        if savefig:
            plt.savefig("plot_gradient_and_potential.png")


    def phi(self, z):
        a, b = [self.param_dict[key] for key in ["a", "b"]]
        c = np.abs(a-b)/np.sqrt(4*a*b)
        return 0.5*((a+b)*self.sigma_1(z+c)+(a-b))

    def phi_alpha(self, z):
        vect_bump = np.vectorize(self.bump_function, otypes = [float])
        bump_h = vect_bump(z/self.r_alpha)
        phi = self.phi(z-self.d_alpha)
        return bump_h*phi

    def get_n_ij(self, i_node, j_neighbors, norm_q_ij):
        q_ij_lon = self.observation.platform_state.lon.m[j_neighbors]- self.observation.platform_state.lon.m[i_node]
        q_ij_lat = self.observation.platform_state.lat.m[j_neighbors]- self.observation.platform_state.lat.m[i_node]
        q_ij = np.vstack((q_ij_lon, q_ij_lat))
        return q_ij/np.sqrt(1+self.param_dict["epsilon"]*norm_q_ij.T) # columns are per neighbor
        #TODO check dimension mismatch: get a 3D array not expected

    def get_u_i(self, node_i):
        # TODO: problem when platforms become disconnected: adjacency mat is 0
        neighbors_idx = np.argwhere(self.adjacency_mat[node_i,:]>0).flatten()
        n_ij = self.get_n_ij(i_node=node_i, j_neighbors=neighbors_idx, norm_q_ij=self.adjacency_mat[node_i, neighbors_idx])
        q_ij_sigma_norm = self.sigma_norm(z_norm=self.adjacency_mat[node_i, neighbors_idx])
        gradient_term = np.sum(self.phi_alpha(z=q_ij_sigma_norm)*n_ij, axis=1)
        u_i = gradient_term*self.dt_in_s # integrate since we have a velocity input
        return PlatformAction(np.linalg.norm(u_i, ord=2)
            / self.u_max_mps,  # scale in % of max u
            direction=np.arctan2(u_i[1], u_i[0]),)
        # neighbors_iter = self.G_proximity.neighbors(node_i)
        # while True:
        #     try:
        #         # get neighbor
        #         node_j = next(neighbors_iter)
        #         dist = self.G_proximity[node_i][node_j]["weight"].m
        #         #phi_alpha_norm
        #     except StopIteration:
        #         # if StopIteration is raised, break from loop
        #         break

class MultiAgentPlanner(HJReach2DPlanner):
    def __init__(
        self,
        problem: NavigationProblem,
        multi_agent_settings: Dict,
        specific_settings: Optional[Dict] = None,
    ):
        super().__init__(problem, specific_settings=specific_settings)
        self.multi_agent_settings = multi_agent_settings
        self.platform_dict = specific_settings["platform_dict"]

    def get_action_set_HJ_naive(self, observation: ArenaObservation) -> PlatformActionSet:
        action_list = []
        for k in range(len(observation)):
            action_list.append(super().get_action(observation[k]))
        return PlatformActionSet(action_list)

    def get_action_HJ_decentralized_reactive_control(
        self, observation: ArenaObservation
    ) -> PlatformActionSet:
        action_list = []
        reactive_control = DecentralizedReactiveControl(
            observation=observation,
            param_dict=self.multi_agent_settings["reactive_control"],
            platform_dict=self.platform_dict,
            nb_max_neighbors=2,
        )
        for k in range(len(observation)):
            hj_navigation = super().get_action(observation[k])
            #hj_navigation = PlatformAction(magnitude=0, direction=0)
            reactive_action = reactive_control.get_reactive_control(k, hj_navigation)
            action_list.append(self.to_platform_action_bounds(reactive_action))
        return PlatformActionSet(action_list)

    def get_action_HJ_with_flocking(self, observation: ArenaObservation) -> PlatformActionSet:
        action_list = []
        flocking_control = FlockingControl(observation=observation, param_dict=self.multi_agent_settings["flocking"], platform_dict=self.platform_dict)
        for k in range(len(observation)):
            #hj_navigation = super().get_action(observation[k])
            flocking_action = flocking_control.get_u_i(node_i=k)
            action_list.append(self.to_platform_action_bounds(flocking_action))
        return PlatformActionSet(action_list)    

    def to_platform_action_bounds(self, action: PlatformAction):
        action.direction = action.direction % (2*np.pi)
        if action.magnitude > 1:  # more than 100% of umax
            action.magnitude = 1
        return action
