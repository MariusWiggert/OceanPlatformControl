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
        self.adjacency_mat = observation.graph_obs.complete_adjacency_matrix_in_unit("m")
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

    def to_platform_action_bounds(self, action: PlatformAction):
        action.direction = action.direction % (2*np.pi)
        if action.magnitude > 1:  # more than 100% of umax
            action.magnitude = 1
        return action
