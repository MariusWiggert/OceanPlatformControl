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

        if len(self.config['map']['features']['observer']) > 0:
            with open(f'config/reinforcement_learning/config_GP_for_reinforcement_learning.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.observer = Observer(config['observer'])

    @staticmethod
    def get_observation_space(config):
        map_features = config['map']['features']['ttr_forecast'] + config['map']['features']['ttr_hindcast'] + len(config['map']['features']['observer'])
        if config['map']['flatten']:
            shape = (config['map']['xy_width_points']**2 * map_features, )
        elif map_features > 1:
            shape = (config['map']['xy_width_points'], config['map']['xy_width_points'], map_features)
        else:
            shape = (config['map']['xy_width_points'], config['map']['xy_width_points'])

        map_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=shape,
        )
        return gym.spaces.Tuple([
            gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(config['num_measurements'], 4)
            ), map_space
        ]) if config['num_measurements'] > 0 else map_space

    def get_features_from_state(
        self,
        fc_obs: ArenaObservation,
        hc_obs: ArenaObservation,
        problem: NavigationProblem
    ) -> Tuple[np.ndarray, np.ndarray]:
        # # Step 1: Save measurements and calculate relative measurements
        if self.config['num_measurements'] > 0:
            self.measurements = np.append(self.measurements[1-self.config['num_measurements']:], np.array([[
                fc_obs.platform_state.lon.deg,
                fc_obs.platform_state.lat.deg,
                fc_obs.true_current_at_state.u,
                fc_obs.true_current_at_state.v
            ]]), axis=0)
            measurements = self.measurements - np.repeat(np.array([[1,1,0,0]]), repeats=self.measurements.shape[0], axis=0) * self.measurements[-1]
            repeats = np.repeat(measurements[-1:], repeats=(self.config['num_measurements']-self.measurements.shape[0]), axis=0)
            measurements = np.append(measurements, repeats, axis=0)

        map = np.zeros((self.config['map']['xy_width_points'], self.config['map']['xy_width_points'], 0), dtype='float32')

        # Step 2: Get TTR Map on grid
        if self.config['map']['features']['ttr_forecast']:
            ttr_forecast = self.forecast_planner.interpolate_value_function_in_hours(fc_obs, width_deg=self.config['map']['xy_width_degree'], width=self.config['map']['xy_width_points'])
            map = np.concatenate((map, np.expand_dims(ttr_forecast, axis=2)), axis=2, dtype='float32')
        if self.config['map']['features']['ttr_hindcast']:
            ttr_hindcast = self.hindcast_planner.interpolate_value_function_in_hours(hc_obs, width_deg=self.config['map']['xy_width_degree'], width=self.config['map']['xy_width_points'])
            map = np.concatenate((map, np.expand_dims(ttr_hindcast, axis=2)), axis=2, dtype='float32')

        # Step 3: Get GP Observer on grid
        if len(self.config['map']['features']['observer']) > 0:
            self.observer.observe(fc_obs)
            self.observer.fit()
            gp_map = self.observer.get_data_over_area(
                x_interval=[fc_obs.platform_state.lon.deg - self.config['map']['xy_width_degree'] / 2, fc_obs.platform_state.lon.deg + self.config['map']['xy_width_degree'] / 2],
                y_interval=[fc_obs.platform_state.lat.deg - self.config['map']['xy_width_degree'] / 2, fc_obs.platform_state.lat.deg + self.config['map']['xy_width_degree'] / 2],
                t_interval=[fc_obs.platform_state.date_time, fc_obs.platform_state.date_time],
            ).interp(
                time=np.datetime64(fc_obs.platform_state.date_time.replace(tzinfo=None)),
                lon=np.linspace(fc_obs.platform_state.lon.deg - self.config['map']['xy_width_degree'] / 2, fc_obs.platform_state.lon.deg + self.config['map']['xy_width_degree'] / 2, self.config['map']['xy_width_points']),
                lat=np.linspace(fc_obs.platform_state.lat.deg - self.config['map']['xy_width_degree'] / 2, fc_obs.platform_state.lat.deg + self.config['map']['xy_width_degree'] / 2, self.config['map']['xy_width_points']),
                method='linear'
            )[self.config['map']['features']['observer']].to_array().to_numpy().astype('float32').swapaxes(0,1).swapaxes(1,2)
            map = np.concatenate((map, gp_map), axis=2, dtype='float32')

        if self.config['map']['flatten']:
            map = map.flatten()
        else:
            map = map.squeeze()


        return (measurements, map) if self.config['num_measurements'] > 0 else map
