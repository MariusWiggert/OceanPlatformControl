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
        nb_max_neighbors: int=2,
    ):
        self.nb_max_neighbors = nb_max_neighbors
        self.param_dict = param_dict
        self.adjacency_mat=nx.to_numpy_array(observation.graph_obs.G_complete)
        # argspartition ensures that the all elements before nb_max_neighbors are the smallest elements
        self.g_a, self.g_b = None
        self.observation = observation
        self.mode_a, self.mode_b

    def get_reactive_control(self, pltf_id, hj_optimal_action):
        self._set_constraint_g(pltf_id=pltf_id)
        a, b = self._set_attraction_or_repulsion(self.g_a), self._set_attraction_or_repulsion(self.g_b)
        if a==0 and b==0: # GoToGoal
           return hj_optimal_action
        elif self.g_a > self.param_dict["delta_1"] or self.g_b > self.param_dict["delta_1"]: # achieve connectivity
            return self._compute_potential_force(pltf_id=pltf_id, a=a, b=b)
        else: # maintain connectivity
            return self._compute_potential_force(pltf_id=pltf_id, a=a, b=b) - self.param_dict["delta_2"]*hj_optimal_action

            
    def _compute_potential_force(self, pltf_id, a, b):
        potential_force = -self.param_dict["k_1"]*(a*self._compute_gradient(pltf_id, self.platf_a) + b*self._compute_gradient(pltf_id, self.platf_b))
        return PlatformAction(magnitude=np.linalg.norm(potential_force, ord=2),
                    direction=np.arctan2(potential_force[1], potential_force[0]))

    def _compute_gradient_g(self, pltf_id, d_x_id):
        # units of hj is m/s and rad/s
        return 2*np.array([self.observation[pltf_id].lon.m- self.observation[d_x_id].lon.m,
                           self.observation[pltf_id].lat.m- self.observation[d_x_id].lat.m])


    def _set_constraint_g(self, pltf_id):
        self.pltf_a, self.pltf_b = np.argpartition(self.adjacency_mat[pltf_id,:],self.nb_max_neighbors)[:self.nb_max_neighbors]
        self.g_a = self.adjacency_mat[pltf_id, self.pltf_a]^2 - self.param_dict["communication_thrsld"]
        self.g_b = self.adjacency_mat[pltf_id, self.pltf_b]^2 - self.param_dict["communication_thrsld"]
    
    def _set_attraction_or_repulsion(self, g):
        if g <= self.param_dict["delta_3"]:
            return -1
        elif g >= self.param_dict["delta_2"]:
            return 1
        else: # delta_3 < g < delta_2
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

    def get_action_set_HJ_naive(self, observation: ArenaObservation) -> PlatformActionSet:
        action_list = []
        for k in range(len(observation)):
            action_list.append(super().get_action(observation[k]))
        return PlatformActionSet(action_list)

    def get_action_HJ_decentralized_reactive_control(self, observation: ArenaObservation) -> PlatformActionSet:
        action_list = []
        reactive_control = DecentralizedReactiveControl(observation=observation, 
                          param_dict=self.multi_agent_setting["reactive_control"], nb_max_neighbors=2)
        for k in range(len(observation)):
            hj_navigation = super().get_action(observation[k])
            action_list.append(reactive_control.get_reactive_control(k, hj_navigation))
        return PlatformActionSet(action_list)