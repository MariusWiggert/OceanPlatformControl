import datetime
import math
import time
from typing import Optional

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
    defaul_config = {
        'num_measurements': 5,
        'ttr': {
            'xy_width_degree': 1,
            'xy_width_points': 10,
        },
    }

    def __init__(self, planner: HJReach2DPlanner, config: Optional[dict] = {}, verbose: Optional[bool] = False):
        self.planner = planner
        self.config = self.defaul_config | config
        self.verbose =verbose
        self.measurements = np.zeros(shape=(0, 4))

    @staticmethod
    def get_observation_space(config: Optional[dict] = {}):
        config = OceanFeatureConstructor.defaul_config | config

        number_of_features = (config['num_measurements'] * 4) + (config['ttr']['xy_width_points']**2)

        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(number_of_features,)
        )

    def get_features_from_state(self, observation: ArenaObservation, problem: NavigationProblem) -> np.ndarray:
        # Step 1: Save measurements and calculate relative measurements
        self.measurements = np.append(self.measurements[1-self.config['num_measurements']:], np.array([[
            observation.platform_state.lon.deg,
            observation.platform_state.lat.deg,
            observation.true_current_at_state.u,
            observation.true_current_at_state.v
        ]]), axis=0)
        measurements = self.measurements - np.repeat(np.array([[1,1,0,0]]), repeats=self.measurements.shape[0], axis=0) * self.measurements[-1]
        repeats = np.repeat(measurements[-1:], repeats=(self.config['num_measurements']-self.measurements.shape[0]), axis=0)
        measurements = np.append(measurements, repeats, axis=0)

        # Step 2: Get forecast on grid
        # forecast = observation.forecast_data_source.get_data_over_area(
        #     x_interval=[observation.platform_state.lon.deg-self.config['forecast'][_map_width_degree']/2,observation.platform_state.lon.deg+self.config['forecast_map_width_degree']/2],
        #     y_interval=[observation.platform_state.lat.deg-self.config['forecast_map_width_degree']/2,observation.platform_state.lat.deg+self.config['forecast_map_width_degree']/2],
        #     t_interval=[observation.platform_state.date_time, observation.platform_state.date_time+datetime.timedelta(hours=self.config['forecast_time_width_hours'])],
        #     spatial_resolution=self.config['forecast_map_width_degree']/(self.config['forecast_map_width_points']-2),
        #     temporal_resolution=int(3600*self.config['forecast_time_width_hours']/(self.config['forecast_time_points']-2)),
        # )
        # print(forecast)
        # forecast = forecast.to_array().to_numpy()
        # print(forecast.shape)

        # Step 3: Get TTR Map on grid
        start = time.time()
        ttr_map = self.planner.interpolate_value_function_in_hours_on_grid(observation, width_deg=self.config['ttr']['xy_width_degree'], width=self.config['ttr']['xy_width_points'])
        if self.verbose:
            print(f'OceanFeatureConstructor: Calculate TTR ({time.time() - start:.1f}s)')

        # print(ttr_map)

        return np.concatenate((ttr_map.flatten(), measurements.flatten()))
