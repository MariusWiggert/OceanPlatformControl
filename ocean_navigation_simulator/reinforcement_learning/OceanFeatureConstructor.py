import datetime
import math
import time
from typing import Optional, Tuple

import gym
import numpy as np

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.FeatureConstructor import FeatureConstructor
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem

"""
    Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
    and then convert to a numpy array that the RL model can use.
"""
class  OceanFeatureConstructor(FeatureConstructor):
    def __init__(self, planner: HJReach2DPlanner, config: dict, verbose: Optional[int] = 0):
        self.planner = planner
        self.config = config
        self.verbose =verbose
        self.measurements = np.zeros(shape=(0, 4))

    @staticmethod
    def get_observation_space(config):
        measurement_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(config['num_measurements'], 4)
        )
        ttr_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(config['ttr']['xy_width_points'], config['ttr']['xy_width_points'])
        )
        return gym.spaces.Tuple([measurement_space, ttr_space]) if config['num_measurements'] else ttr_space

    def get_features_from_state(self, observation: ArenaObservation, problem: NavigationProblem) -> Tuple[np.ndarray, np.ndarray]:
        # Step 1: Save measurements and calculate relative measurements
        if self.config['num_measurements'] > 0:
            self.measurements = np.append(self.measurements[1-self.config['num_measurements']:], np.array([[
                observation.platform_state.lon.deg,
                observation.platform_state.lat.deg,
                observation.true_current_at_state.u,
                observation.true_current_at_state.v
            ]]), axis=0)
            measurements = self.measurements - np.repeat(np.array([[1,1,0,0]]), repeats=self.measurements.shape[0], axis=0) * self.measurements[-1]
            repeats = np.repeat(measurements[-1:], repeats=(self.config['num_measurements']-self.measurements.shape[0]), axis=0)
            measurements = np.append(measurements, repeats, axis=0)

        # Step 3: Get TTR Map on grid
        start = time.time()
        ttr_map = self.planner.interpolate_value_function_in_hours(observation, width_deg=self.config['ttr']['xy_width_degree'], width=self.config['ttr']['xy_width_points'])
        # if self.verbose > 0:
            # print(f'OceanFeatureConstructor: Calculate TTR ({time.time() - start:.1f}s)')

        return (measurements, ttr_map) if self.config['num_measurements'] > 0 else ttr_map
