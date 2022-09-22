import datetime
import math
import time
from typing import Optional, Tuple

import gym
import numpy as np
import yaml

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.FeatureConstructor import FeatureConstructor
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.ocean_observer.Observer import Observer

"""
    Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
    and then convert to a numpy array that the RL model can use.
"""
class  OceanFeatureConstructor(FeatureConstructor):
    def __init__(self, forecast_planner: HJReach2DPlanner, hindcast_planner: HJReach2DPlanner, config: dict, verbose: Optional[int] = 0):
        self.forecast_planner = forecast_planner
        self.hindcast_planner = hindcast_planner
        self.config = config
        self.verbose =verbose

        if self.config['num_measurements'] > 0:
            self.measurements = np.zeros(shape=(0, 4))

        if len(self.config['map']['observer']) > 0:
            with open(f'config/reinforcement_learning/config_GP_for_reinforcement_learning.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.observer = Observer(config['observer'])

    @staticmethod
    def get_observation_space(config):
        measurement_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(config['num_measurements'], 4)
        )
        map_features = config['map']['ttr_forecast'] + config['map']['ttr_hindcast'] + len(config['map']['observer'])
        map_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(config['map']['xy_width_points'], config['map']['xy_width_points'], map_features) if map_features>1 else (config['map']['xy_width_points'], config['map']['xy_width_points'])
        )
        return gym.spaces.Tuple([measurement_space, map_space]) if config['num_measurements'] else map_space

    def get_features_from_state(self, observation: ArenaObservation, problem: NavigationProblem) -> Tuple[np.ndarray, np.ndarray]:
        # # Step 1: Save measurements and calculate relative measurements
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

        # Step 2: Get TTR Map on grid
        if self.config['map']['ttr_forecast']:
            ttr_foreacst = self.forecast_planner.interpolate_value_function_in_hours(observation, width_deg=self.config['map']['xy_width_degree'], width=self.config['map']['xy_width_points'])
        if self.config['map']['ttr_hin']:
            ttr_foreacst = self.forecast_planner.interpolate_value_function_in_hours(observation, width_deg=self.config['map']['xy_width_degree'], width=self.config['map']['xy_width_points'])

        # ttr_map = np.expand_dims(ttr_map, axis=2)

        # Step 3: Get GP Observer on grid
        if len(self.config['map']['observer']) > 0:
            self.observer.observe(observation)
            self.observer.fit()
            gp_map = self.observer.get_data_over_area(
                x_interval=[observation.platform_state.lon.deg - self.config['map']['xy_width_degree']/2, observation.platform_state.lon.deg + self.config['map']['xy_width_degree']/2],
                y_interval=[observation.platform_state.lat.deg - self.config['map']['xy_width_degree']/2, observation.platform_state.lat.deg + self.config['map']['xy_width_degree']/2],
                t_interval=[observation.platform_state.date_time, observation.platform_state.date_time],
            ).interp(
                time=np.datetime64(observation.platform_state.date_time.replace(tzinfo=None)),
                lon=np.linspace(observation.platform_state.lon.deg - self.config['map']['xy_width_degree'] / 2, observation.platform_state.lon.deg + self.config['map']['xy_width_degree'] / 2, self.config['map']['xy_width_points']),
                lat=np.linspace(observation.platform_state.lat.deg - self.config['map']['xy_width_degree'] / 2, observation.platform_state.lat.deg + self.config['map']['xy_width_degree'] / 2, self.config['map']['xy_width_points']),
                method='linear'
            )[self.config['map']['observer']].to_array().to_numpy().astype('float32').swapaxes(0,1).swapaxes(1,2)
            map = np.concatenate((ttr_map, gp_map), axis=2)
        else:
            map = ttr_map

        return (measurements, ttr_map) if self.config['num_measurements'] > 0 else ttr_map
