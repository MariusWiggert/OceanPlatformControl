import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import scipy.integrate as integrate
import xarray as xr

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.controllers.DecentralizedReactiveControl import (
    DecentralizedReactiveControl,
)
from ocean_navigation_simulator.controllers.Flocking import (  # RelaxedFlockingControl,; FlockingControl2,
    FlockingControl,
    FlockingControlVariant,
)

from ocean_navigation_simulator.controllers.MultiAgentOptimizationBased import (
    MultiAgentOptim,
    CentralizedMultiAgentMPC,
)

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJPlannerBase,
    HJReach2DPlanner,
)
from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.MultiAgent import MultiAgent
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
from ocean_navigation_simulator.utils import units


class MultiAgentPlanner(HJReach2DPlanner):
    """
    Base Class for all the multi-agent computations, to try to maintain connectivity and avoid
    collisions. Child Class of HJReach2dPlanner to run HJ as a navigation function for every platforms
    using multi-time reachability
    """

    def __init__(
        self,
        problem: NavigationProblem,
        specific_settings: Dict,
    ):
        super().__init__(problem, specific_settings=specific_settings["hj_specific_settings"])
        self.multi_agent_settings = specific_settings
        self.platform_dict = problem.platform_dict
        self.controller_type = specific_settings["high_level_ctrl"]
        self.hj = super()

    def get_action(self, observation: ArenaObservation) -> PlatformActionSet:
        if self.controller_type == "hj_naive":
            return self._get_action_hj_naive(observation=observation)
        elif self.controller_type == "reactive_control":
            return self._get_action_hj_decentralized_reactive_control(observation=observation)
        elif self.controller_type == "flocking":
            return self._get_action_hj_with_flocking(observation=observation)
        elif self.controller_type == "multi_ag_optimizer":
            return self._get_action_hj_with_multi_ag_optim(observation=observation)
        elif self.controller_type == "pred_safety_filter":
            return self._get_action_hj_with_pred_safety_filter(observation=observation)
        elif self.controller_type == "centralized_mpc":
            return self._get_action_hj_with_centralized_mpc(observation=observation)
        else:
            raise ValueError(
                "the controller specified in the config is not implemented or must \
            not be empty"
            )

    def _get_action_hj_naive(
        self, observation: ArenaObservation
    ) -> Tuple[PlatformActionSet, float, float]:
        """Obtain pure HJ action for each platform, without multi-agent constraints consideration

        Args:
            observation (ArenaObservation): Arena Observation with states, currents etc.

        Returns:
            PlatformActionSet: A set of platform actions
        """
        action_list = []
        times_list = []
        ctrl_correction_angle = 0
        for k in range(len(observation)):
            start = time.time()
            action_list.append(self.hj.get_action(observation[k]))
            times_list.append(time.time() - start)
        return (PlatformActionSet(action_list), ctrl_correction_angle, sum(times_list))

    def get_hj_ttr_values(self, observation: ArenaObservation) -> np.ndarray:
        """Get ttr values: Important thing to watch out:
          time of observation point should be within reach time otherwsie
          throws an error !!

        Args:
            observation (ArenaObservation): _description_

        Returns:
            np.ndarray: _description_
        """
        ttr_values_list = []
        for pltf_id in range(len(observation)):
            point = observation.platform_state[pltf_id].to_spatio_temporal_point()
            ttr_values_list.append(
                self.hj.interpolate_value_function_in_hours(point=point)
            )  # interpolate TTR map value
        return np.array(ttr_values_list)

    def _get_action_hj_decentralized_reactive_control(
        self, observation: ArenaObservation
    ) -> Tuple[PlatformActionSet, float, float]:
        """Reactive control for multi-agent, with HJ as navigation function

        Args:
            observation (ArenaObservation): Arena Observation with states, currents
                                            and Graph Observations

        Returns:
            PlatformActionSet: A set of platform actions computed using reactive control and HJ
        """
        action_list = []
        times_list = []
        reactive_control_correction_angle = []
        if self.multi_agent_settings["reactive_control_config"]["mix_ttr_and_euclidean"]:
            self.hj.get_action(
                observation[0]
            )  # intialize the planner, needed first time before getting the ttr values
            ttr_values = self.get_hj_ttr_values(observation=observation)
        else:
            ttr_values = None
        reactive_control = DecentralizedReactiveControl(
            observation=observation,
            param_dict=self.multi_agent_settings["reactive_control_config"],
            platform_dict=self.platform_dict,
            ttr_values_arr=ttr_values,
            nb_max_neighbors=2,
        )
        for k in range(len(observation)):
            start = time.time()
            hj_navigation = self.hj.get_action(observation[k])
            reactive_action = reactive_control.get_reactive_control(k, hj_navigation)
            times_list.append(time.time() - start)
            action_list.append(self.to_platform_action_bounds(reactive_action))
            # compute the reactive correction angle to optimal input as proxy for energy consumption
            # map angle to [-pi,pi] and take the absolute value
            reactive_control_correction_angle.append(
                abs(math.remainder(reactive_action.direction - hj_navigation.direction, math.tau))
            )
        return (
            PlatformActionSet(action_list),
            max(reactive_control_correction_angle),
            sum(times_list),
        )

    def _get_action_hj_with_flocking(
        self, observation: ArenaObservation
    ) -> Tuple[PlatformActionSet, float, float]:
        """multi-agent control input based on flocking and using HJ to reach the target

        Args:
            observation (ArenaObservation): Arena Observation with states, currents
                                            and Graph Observations

        Returns:
            PlatformActionSet: A set of platform actions computed using flocking and HJ
        """
        action_list = []
        flocking_correction_angle = []
        flocking_control = FlockingControl(
            observation=observation,
            param_dict=self.multi_agent_settings["flocking_config"],
            platform_dict=self.platform_dict,
        )
        times_list = []
        for k in range(len(observation)):
            start = time.time()
            hj_navigation = self.hj.get_action(observation[k])
            flocking_action = flocking_control.get_u_i(node_i=k, hj_action=hj_navigation)
            times_list.append(time.time() - start)
            action_list.append(self.to_platform_action_bounds(flocking_action))
            # compute the flocking correction angle to optimal input as proxy for energy consumption
            # map angle to [-pi,pi] and take the absolute value
            flocking_correction_angle.append(
                abs(math.remainder(flocking_action.direction - hj_navigation.direction, math.tau))
            )
        return (PlatformActionSet(action_list), max(flocking_correction_angle), sum(times_list))

    def _get_action_hj_with_multi_ag_optim(
        self, observation: ArenaObservation
    ) -> Tuple[PlatformActionSet, float, float]:

        horizon = self.multi_agent_settings["multi_ag_optim"]["optim_horizon"]
        start = time.time()
        hj_actions = [
            self.hj.get_action_over_horizon(
                observation=observation[k], horizon=horizon, dt_in_sec=self.platform_dict["dt_in_s"]
            )
            for k in range(len(observation))
        ]
        hj_solve_time = time.time() - start
        # Obtain a list of length corresponding to the horizon
        # Each element in the list is an action array with first dim: #platforms
        # and second dim the [x y] propulsions
        hj_in_xy_propulsion = [
            np.array(
                [hj_actions_of_pltf[step].to_xy_propulsion() for hj_actions_of_pltf in hj_actions]
            )
            for step in range(horizon - 1)
        ]
        multi_ag_optim = MultiAgentOptim(
            observation=observation,
            param_dict=self.multi_agent_settings["multi_ag_optim"],
            platform_dict=self.platform_dict,
        )
        opt_actions, time_solver = multi_ag_optim.get_next_control_for_all_pltf(
            hj_in_xy_propulsion, hj_horizon_full=True
        )
        opt_actions = [
            self.to_platform_action_bounds(action, scale_magnitude=False) for action in opt_actions
        ]
        correction_angles = [
            abs(math.remainder(optimized_input.direction - hj_input.direction, math.tau))
            for optimized_input, hj_input in zip(
                opt_actions, [hj_action[0] for hj_action in hj_actions]
            )
        ]
        return (
            PlatformActionSet(action_set=opt_actions),
            max(correction_angles),
            time_solver + hj_solve_time,
        )

    def _get_action_hj_with_pred_safety_filter(
        self, observation: ArenaObservation
    ) -> Tuple[PlatformActionSet, float, float]:

        start = time.time()
        hj_actions = [self.hj.get_action(observation[k]) for k in range(len(observation))]
        hj_solve_time = time.time() - start
        hj_xy_propulsion_arr = np.array([hj_input.to_xy_propulsion() for hj_input in hj_actions])
        multi_ag_optim = MultiAgentOptim(
            observation=observation,
            param_dict=self.multi_agent_settings["multi_ag_optim"],
            platform_dict=self.platform_dict,
        )
        opt_actions, time_solver = multi_ag_optim.get_next_control_for_all_pltf(
            hj_xy_propulsion_arr,
            hj_horizon_full=False,
        )
        opt_actions = [self.to_platform_action_bounds(action) for action in opt_actions]
        correction_angles = [
            abs(math.remainder(optimized_input.direction - hj_input.direction, math.tau))
            for optimized_input, hj_input in zip(opt_actions, hj_actions)
        ]
        return (
            PlatformActionSet(action_set=opt_actions),
            max(correction_angles),
            time_solver + hj_solve_time,
        )

    def _get_action_hj_with_centralized_mpc(
        self, observation: ArenaObservation
    ) -> Tuple[PlatformActionSet, float, float]:
        start = time.time()
        start = time.time()
        hj_actions = [self.hj.get_action(observation[k]) for k in range(len(observation))]
        hj_solve_time = time.time() - start
        hj_xy_propulsion_arr = np.array([hj_input.to_xy_propulsion() for hj_input in hj_actions])
        multi_ag_mpc = CentralizedMultiAgentMPC(
            observation=observation,
            param_dict=self.multi_agent_settings["multi_ag_mpc"],
            platform_dict=self.platform_dict,
        )
        opt_actions, time_solver = multi_ag_mpc.get_next_control_for_all_pltf(
            hj_xy_propulsion_arr,
        )
        opt_actions = [
            self.to_platform_action_bounds(action, scale_magnitude=False) for action in opt_actions
        ]
        correction_angles = [
            abs(math.remainder(optimized_input.direction - hj_input.direction, math.tau))
            for optimized_input, hj_input in zip(opt_actions, hj_actions)
        ]
        return (
            PlatformActionSet(action_set=opt_actions),
            max(correction_angles),
            time_solver + hj_solve_time,
        )

    def to_platform_action_bounds(
        self, action: PlatformAction, scale_magnitude: Optional[bool] = True
    ) -> PlatformAction:
        """Bound magnitude to 0-1 of u_max and direction between [0, 2pi[

        Args:
            action (PlatformAction)

        Returns:
            PlatformAction: scaled w.r.t u_max
        """
        action.direction = action.direction % (2 * np.pi)
        if scale_magnitude:
            action.magnitude = max(min(action.magnitude, 1), 1)
        else:
            action.magnitude = min(action.magnitude, 1)
        return action
