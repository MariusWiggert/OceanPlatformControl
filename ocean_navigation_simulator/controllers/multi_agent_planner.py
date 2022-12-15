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
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(
            unit="m", graph_type="complete"
        )
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
            and self.g_b > -self.param_dict["delta_1"] ** 2
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
        grad = 2 * np.array(
            [
                self.observation[pltf_id].platform_state.lon.m
                - self.observation[d_x_id].platform_state.lon.m,
                self.observation[pltf_id].platform_state.lat.m
                - self.observation[d_x_id].platform_state.lat.m,
            ]
        )
        # return normalized gradient
        return grad / np.linalg.norm(grad, ord=2)

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
        self.adjacency_mat = observation.graph_obs.adjacency_matrix_in_unit(
            unit="m", graph_type="communication"
        )  # get adjacency
        self.u_max_mps = platform_dict["u_max_in_mps"]
        self.dt_in_s = platform_dict["dt_in_s"]
        self.r_alpha = self.sigma_norm(self.param_dict["interaction_range"])
        self.d_alpha = self.sigma_norm(self.param_dict["ideal_distance"])
        self.a, self.b = [self.param_dict[key] for key in ["a", "b"]]
        # self.c = np.abs(self.a - self.b) / np.sqrt(4 * self.a * self.b)
        self.c = -self.a / np.sqrt(2 * self.a * self.b + self.b**2)
        self.vect_bump_f = np.vectorize(self.bump_function, otypes=[float])

    def sigma_norm(self, z_norm):
        val = self.param_dict["epsilon"] * (z_norm**2)
        return 1 / self.param_dict["epsilon"] * (np.sqrt(1 + val) - 1)

    def sigma_1(self, z):
        return z / np.sqrt(1 + z**2)

    def bump_function(self, z):
        # maybe can vectorize it directly with np.where(condition, val if true, val if false) but we have 3 different vals to assign for
        # 3 different conditions::
        # z = np.where(np.logical_and(z>=0, z < self.param_dict["h"]), 1, z)
        if 0 <= z < self.param_dict["h"]:
            return 1
        elif self.param_dict["h"] <= z < 1:
            arg = np.pi * (z - self.param_dict["h"]) / (1 - self.param_dict["h"])
            return 0.5 * (1 + np.cos(arg))
        else:
            return 0

    def plot_psi_and_phi_alpha(self, step: Optional[int] = 100, savefig: Optional[bool] = False):
        z_range = np.arange(start=0, stop=self.r_alpha, step=step)
        phi_alpha = [self.phi_alpha(z=z) for z in z_range]
        psi_alpha = [integrate.quad(self.phi_alpha, self.d_alpha, z)[0] for z in z_range]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(z_range, phi_alpha)
        ax1.set_ylabel(r"$\phi_{\alpha}$")
        ax1.set_xticks([self.d_alpha, self.r_alpha])
        ax1.set_xticklabels([r"$d_{\alpha}$", r"$r_{\alpha}$"])
        ax1.grid(axis="both", linestyle="--")
        ax2.plot(z_range, psi_alpha)
        ax2.set_ylabel(r"$\psi_{\alpha}$")
        ax2.set_xticks([self.d_alpha, self.r_alpha])
        ax2.set_xticklabels([r"$d_{\alpha}$", r"$r_{\alpha}$"])
        ax2.set_xlabel(r"$\Vert z \Vert_{\sigma}$")
        ax2.grid(axis="both", linestyle="--")
        if savefig:
            plt.savefig("plot_gradient_and_potential.png")

    def phi(self, z):
        return 0.5 * (
            (self.a + self.b) * self.sigma_1(z + self.c) + self.a
        )  # 0.5 * ((self.a + self.b) * self.sigma_1(z + self.c) + (self.a - self.b))

    def phi_alpha(self, z):
        bump_h = self.vect_bump_f(z / self.r_alpha)
        phi = self.phi(z - self.d_alpha)
        return bump_h * phi

    def get_n_ij(self, i_node, j_neighbors, norm_q_ij):
        q_ij_lon = (
            self.observation.platform_state.lon.m[j_neighbors]
            - self.observation.platform_state.lon.m[i_node]
        )
        q_ij_lat = (
            self.observation.platform_state.lat.m[j_neighbors]
            - self.observation.platform_state.lat.m[i_node]
        )
        q_ij = np.vstack((q_ij_lon, q_ij_lat))
        return q_ij / np.sqrt(
            1 + self.param_dict["epsilon"] * (norm_q_ij.T) ** 2
        )  # columns are per neighbor
        # TODO check dimension mismatch: get a 3D array not expected

    def get_velocity_diff_array(self, node_i, neighbors_idx, sign_only: bool = False):
        velocities = np.array(self.observation.platform_state.velocity)
        # Enforce good dimensions with reshape (velocities u and v as rows)
        velocity_diff = velocities[:, node_i].reshape(2, 1) - velocities[:, neighbors_idx].reshape(
            2, neighbors_idx.size
        )
        if sign_only:
            return np.sign(velocity_diff)
        else:
            return velocity_diff

    def get_aij(self, q_ij_sigma):
        return self.vect_bump_f(q_ij_sigma / self.r_alpha)

    def get_u_i(self, node_i: int, hj_action: PlatformAction):
        neighbors_idx = np.argwhere(self.adjacency_mat[node_i, :] > 0).flatten()
        if not neighbors_idx.size:  # no neighbors
            return hj_action
        else:
            n_ij = self.get_n_ij(
                i_node=node_i,
                j_neighbors=neighbors_idx,
                norm_q_ij=self.adjacency_mat[node_i, neighbors_idx],
            )
            q_ij_sigma_norm = self.sigma_norm(z_norm=self.adjacency_mat[node_i, neighbors_idx])
            gradient_term = np.sum(self.phi_alpha(z=q_ij_sigma_norm) * n_ij, axis=1)
            velocity_match = np.matmul(
                self.get_aij(q_ij_sigma=q_ij_sigma_norm),
                self.get_velocity_diff_array(
                    node_i=node_i, neighbors_idx=neighbors_idx, sign_only=True
                ).T,
            )  # sum over neighbors where velocity is a vector (u,v)
            # p_i = np.array([self.observation.platform_state[node_i].velocity.u.mps,
            #         self.observation.platform_state[node_i].velocity.v.mps]) # platform i velocity
            # p_r = np.array([np.cos(hj_action.direction)*hj_action.magnitude*self.u_max_mps,
            #         np.sin(hj_action.direction)*hj_action.magnitude*self.u_max_mps]) # reference velocity
            u_i = (
                0.5 * (gradient_term + velocity_match) * self.dt_in_s
            )  # - self.param_dict["hj_factor"]*(p_i-p_r)# integrate since we have a velocity input
            u_i_scaled = PlatformAction(
                min(np.linalg.norm(u_i, ord=2) / self.u_max_mps, 1),  # scale in % of max u, bounded
                direction=np.arctan2(u_i[1], u_i[0]),
            )

            return u_i_scaled + hj_action.scaling(self.param_dict["hj_factor"])
        # return u_i_scaled
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

    def get_action_HJ_naive(self, observation: ArenaObservation) -> PlatformActionSet:
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
            # hj_navigation = PlatformAction(magnitude=0, direction=0)
            reactive_action = reactive_control.get_reactive_control(k, hj_navigation)
            action_list.append(self.to_platform_action_bounds(reactive_action))
        return PlatformActionSet(action_list)

    def get_action_HJ_with_flocking(self, observation: ArenaObservation) -> PlatformActionSet:
        action_list = []
        flocking_control = FlockingControl(
            observation=observation,
            param_dict=self.multi_agent_settings["flocking"],
            platform_dict=self.platform_dict,
        )
        for k in range(len(observation)):
            hj_navigation = super().get_action(observation[k])
            flocking_action = flocking_control.get_u_i(node_i=k, hj_action=hj_navigation)
            action_list.append(self.to_platform_action_bounds(flocking_action))
        return PlatformActionSet(action_list)

    def to_platform_action_bounds(self, action: PlatformAction):
        action.direction = action.direction % (2 * np.pi)
        if action.magnitude > 1:  # more than 100% of umax
            action.magnitude = 1
        return action
