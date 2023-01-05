import math
import multiprocessing as mp
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
from ocean_navigation_simulator.ocean_observer.Observer import Observer
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

    def get_hj_ttr_values(self, observation: ArenaObservation) -> np.ndarray:
        ttr_values_list = []
        for pltf_id in range(len(observation)):
            point = observation.platform_state[pltf_id].to_spatio_temporal_point()
            ttr_values_list.append(
                super().interpolate_value_function_in_hours(point=point)
            )  # interpolate TTR map value
        return np.array(ttr_values_list)

    def get_action_HJ_decentralized_reactive_control(
        self, observation: ArenaObservation
    ) -> PlatformActionSet:
        """Reactive control for multi-agent, with HJ as navigation function

        Args:
            observation (ArenaObservation): Arena Observation with states, currents
                                            and Graph Observations

        Returns:
            PlatformActionSet: A set of platform actions computed using reactive control and HJ
        """
        action_list = []
        reactive_control_correction_angle = []
        reactive_control = DecentralizedReactiveControl(
            observation=observation,
            param_dict=self.multi_agent_settings["reactive_control"],
            platform_dict=self.platform_dict,
            ttr_values_arr=self.get_hj_ttr_values(observation=observation),
            nb_max_neighbors=2,
        )
        for k in range(len(observation)):
            hj_navigation = super().get_action(observation[k])
            reactive_action = reactive_control.get_reactive_control(k, hj_navigation)
            action_list.append(self.to_platform_action_bounds(reactive_action))
            # compute the reactive correction angle to optimal input as proxy for energy consumption
            # map angle to [-pi,pi] and take the absolute value
            reactive_control_correction_angle.append(
                abs(math.remainder(reactive_action.direction - hj_navigation.direction, math.tau))
            )
        return PlatformActionSet(action_list), max(reactive_control_correction_angle)

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
        flocking_control = FlockingControl(
            observation=observation,
            param_dict=self.multi_agent_settings["flocking"],
            platform_dict=self.platform_dict,
        )
        for k in range(len(observation)):
            hj_navigation = super().get_action(observation[k])
            flocking_action = flocking_control.get_u_i(node_i=k, hj_action=hj_navigation)
            action_list.append(self.to_platform_action_bounds(flocking_action))
            # compute the flocking correction angle to optimal input as proxy for energy consumption
            # map angle to [-pi,pi] and take the absolute value
            flocking_correction_angle.append(
                abs(math.remainder(flocking_action.direction - hj_navigation.direction, math.tau))
            )
        return PlatformActionSet(action_list), max(flocking_correction_angle)

    def to_platform_action_bounds(self, action: PlatformAction) -> PlatformAction:
        """Bound magnitude to 0-1 of u_max and direction between [0, 2pi[

        Args:
            action (PlatformAction)

        Returns:
            PlatformAction: scaled w.r.t u_max
        """
        action.direction = action.direction % (2 * np.pi)
        action.magnitude = max(min(action.magnitude, 1), 1)
        return action
