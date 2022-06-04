import math

import gym
import numpy as np

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.FeatureConstructors import FeatureConstructor
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem

"""
Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
and then convert to a numpy array that the RL model can use.
"""


class  DoubleGyreFeatureConstructor(FeatureConstructor):
    def get_observation_space(self):
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1,)
        )

    def get_features_from_state(self, obs: ArenaObservation, problem: NavigationProblem) -> np.ndarray:
        """
        Converts the observation to use relative positions
        Args:
            problem: class containing information about RL problem (end region, start state, etc.)
            obs: current platform observation

        Returns:
            numpy array containing relative lat pos, relative lon pos, elapsed time, u_curr, v_curr
        """
        lon_diff = problem.end_region.lon.deg - obs.platform_state.lon.deg
        lat_diff = problem.end_region.lat.deg - obs.platform_state.lat.deg
        distance = obs.platform_state.distance(problem.end_region)
        time_elapsed = (obs.platform_state.date_time - problem.start_state.date_time).total_seconds()

        return np.array([math.atan2(lat_diff, lon_diff)], dtype=np.float32)
        # return np.array([time_elapsed, math.atan2(lat_diff, lon_diff), distance], dtype=np.float32)
        # return np.array([obs.platform_state.lon.deg, obs.platform_state.lat.deg], dtype=np.float32)
