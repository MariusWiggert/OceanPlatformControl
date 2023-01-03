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
from ocean_navigation_simulator.controllers.Flocking import (
    FlockingControl,
    RelaxedFlockingControl,
    FlockingControl2,
)
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.environment.MultiAgent import MultiAgent
import scipy.integrate as integrate
import math

class DecentralizedReactiveControl:
    """Implementation of reactive control for the multi-agent scheme 
    with the rules specified in
    https://repository.upenn.edu/cgi/viewcontent.cgi?article=1044&context=meam_papers
    """
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

    def get_reactive_control(self, pltf_id: int, hj_optimal_action: PlatformAction)-> PlatformAction:
        """Obtain the reactive control input for a given platform, taking into account the position 
        of the neighboring platforms and the navigation function, defined as the HJ time-optimal
        control input

        Args:
            pltf_id (int): the platform for which the control input is computed (index i in the paper)
            hj_optimal_action (PlatformAction): the time-optimal control input for this platform

        Returns:
            PlatformAction: the control input to apply to the platform
        """
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

    def _compute_potential_force(self, pltf_id: int, a: int, b: int) -> PlatformAction:
        """Compute the input -k1*(a*d/dx^i*g^a + b*d/dx^i*g^b)

        Args:
            pltf_id (int): the platform for which the control input is computed (index i in the paper)
            a (int): The constant for constraint of neighbor a of this platform 
                    (defining repulsion or attraction)
            b (int): The constant for constraint of neighbor b of this platform 
                    (defining repulsion or attraction)

        Returns:
            PlatformAction: pure reactive input
        """
        potential_force = -self.param_dict["k_1"] * (
            a * self._compute_gradient_g(pltf_id, self.pltf_a)
            + b * self._compute_gradient_g(pltf_id, self.pltf_b)
        )
        return PlatformAction(
            magnitude=np.linalg.norm(potential_force, ord=2)
            / self.u_max_mps,  # scale in % of max u
            direction=np.arctan2(potential_force[1], potential_force[0]),
        )

    def _compute_gradient_g(self, pltf_id: int, d_x_id: int) -> np.ndarray:
        """Computes the gradient of the constraint function g
        
        Args:
            pltf_id (int): the platform for which the control input is computed (index i in the paper)
            d_x_id (int): constraint of platform i w.r.t a or b (index of the neighboring platform)

        Returns:
            np.ndarray: normalized gradient dg^(a,b)/dx^i
        """
        grad = 2 * np.array(
            [
                self.observation[pltf_id].platform_state.lon.m
                - self.observation[d_x_id].platform_state.lon.m,
                self.observation[pltf_id].platform_state.lat.m
                - self.observation[d_x_id].platform_state.lat.m,
            ]
        )
        return grad / np.linalg.norm(grad, ord=2)

    def _set_constraint_g(self, pltf_id: int):
        """Compute the constraint function g for the neighboring platforms a and b
        of platform i (defined by platf_id for which the reactive control input is
        computed)

        Args:
            pltf_id (int): the platform for which the control input is computed (index i in the paper)
        """
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

    def _set_attraction_or_repulsion(self, g: float)-> int:
        """ Set attraction or repulsive behavior w.r.t to
        neighbor a or b, given by the circular constraint function g
        The thresholds delta are squared to match the constraint function
        form (x_i - x_k)^2 + (y_i - y_k)^2 - r_k^2 
        Args:
            g (float): constraint function g^a or g^b

        Returns:
            int: the computed parameter a or b
        """
        if g <= -self.param_dict["delta_3"] ** 2:
            return -1
        elif g >= -self.param_dict["delta_2"] ** 2:
            return 1
        else:  # delta_3 < g < delta_2
            return 0

class MultiAgentPlanner(HJReach2DPlanner):
    """
    Base Class for all the multi-agent computations, to try to maintain connectivity and avoid 
    collisions. Child Class of HJReach2dPlanner to run HJ as a navigation function for every platforms
    using multi-time reachability
    """
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
        """Obtain pure HJ action for each platform, without multi-agent constraints consideration

        Args:
            observation (ArenaObservation): Arena Observation with states, currents etc.

        Returns:
            PlatformActionSet: A set of platform actions
        """
        action_list = []
        for k in range(len(observation)):
            action_list.append(super().get_action(observation[k]))
        return PlatformActionSet(action_list)

    def get_action_HJ_decentralized_reactive_control(
        self, observation: ArenaObservation
    ) -> PlatformActionSet:
        """ Reactive control for multi-agent, with HJ as navigation function

        Args:
            observation (ArenaObservation): Arena Observation with states, currents
                                            and Graph Observations 

        Returns:
            PlatformActionSet: A set of platform actions computed using reactive control and HJ
        """
        action_list = []
        reactive_control = DecentralizedReactiveControl(
            observation=observation,
            param_dict=self.multi_agent_settings["reactive_control"],
            platform_dict=self.platform_dict,
            nb_max_neighbors=2,
        )
        for k in range(len(observation)):
            hj_navigation = super().get_action(observation[k])
            reactive_action = reactive_control.get_reactive_control(k, hj_navigation)
            action_list.append(self.to_platform_action_bounds(reactive_action))
        return PlatformActionSet(action_list)

    def get_action_HJ_with_flocking(self, observation: ArenaObservation) -> PlatformActionSet:
        """multi-agent control input based on flocking and using HJ to reach the target

        Args:
            observation (ArenaObservation): Arena Observation with states, currents
                                            and Graph Observations 

        Returns:
            PlatformActionSet: A set of platform actions computed using flocking and HJ
        """
        action_list = []
        flocking_correction_angle = []
        # flocking_control = FlockingControl(
        #     observation=observation,
        #     param_dict=self.multi_agent_settings["flocking"],
        #     platform_dict=self.platform_dict,
        # )
        flocking_control = FlockingControl(
            observation=observation,
            param_dict=self.multi_agent_settings["flocking"],
            platform_dict=self.platform_dict,
        )
        for k in range(len(observation)):
            hj_navigation = super().get_action(observation[k])
            point = observation[k].platform_state.to_spatio_temporal_point()
            #val = super().interpolate_value_function_in_hours(point=point) # interpolate TTR map value
            flocking_action = flocking_control.get_u_i(node_i=k, hj_action=hj_navigation)
            action_list.append(self.to_platform_action_bounds(flocking_action))
            # compute the flocking correction angle to optimal input as proxy for energy consumption 
            # map angle to [-pi,pi] and take the absolute value
            flocking_correction_angle.append(abs(math.remainder(flocking_action.direction - hj_navigation.direction, math.tau)))
        return PlatformActionSet(action_list), max(flocking_correction_angle)

    def to_platform_action_bounds(self, action: PlatformAction)->PlatformAction:
        """Bound magnitude to 0-1 of u_max and direction between [0, 2pi[

        Args:
            action (PlatformAction)

        Returns:
            PlatformAction: scaled w.r.t u_max
        """
        action.direction = action.direction % (2 * np.pi)
        action.magnitude = min(action.magnitude,1)
        return action
