import datetime
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

        if self.config['measurements'] > 0:
            self.measurements = np.zeros(shape=(0, 4))

        if len(self.config['local_map']['features']['observer']['variables']) > 0:
            with open(f'config/reinforcement_learning/config_GP_for_reinforcement_learning.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.observer = Observer(config['observer'])

    @staticmethod
    def get_observation_space(config):
        features = []

        # # Step 1: Raw Measurements
        if config['measurements']:
            features += [gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(config['measurements'], 4),
                dtype='float32',
            )]

        # Step 2: Local Map
        if config['local_map']:
            map_features = (config['local_map']['features']['ttr_forecast']
                           + config['local_map']['features']['ttr_hindcast']
                           + len(config['local_map']['features']['observer']['variables']))
            if map_features > 1:
                shape = (config['local_map']['xy_width_points'], config['local_map']['xy_width_points'], map_features)
            else:
                shape = (config['local_map']['xy_width_points'], config['local_map']['xy_width_points'])
            features += [gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=shape,
                dtype='float32',
            )]

        # Step 3: Global Map
        if config['global_map']:
            map_features = (config['global_map']['features']['ttr_forecast']
                           + config['global_map']['features']['ttr_hindcast']
                           + len(config['global_map']['features']['observer']['variables']))
            if map_features > 1:
                shape = (config['global_map']['xy_width_points'], config['global_map']['xy_width_points'], map_features)
            else:
                shape = (config['global_map']['xy_width_points'], config['global_map']['xy_width_points'])
            features += [gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=shape,
                dtype='float32',
            )]

        # Step 4: Meta Information
        if config['meta']:
            meta_features = len(config['meta'])
            features += [gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(meta_features,),
                dtype='float32',
            )]

        return gym.spaces.Tuple([f for f in features]) if len(features) > 1 else features[0]

    def get_features_from_state(
        self,
        fc_obs: ArenaObservation,
        hc_obs: ArenaObservation,
        problem: NavigationProblem
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = ()

        # # Step 1: Raw Measurements
        if self.config['measurements']:
            features += (self.get_measurements_from_state(fc_obs, hc_obs, problem, self.config['measurements']),)

        # Step 2: Local Map
        if self.config['local_map']:
            features += (self.get_map_from_state(fc_obs, hc_obs, problem, self.config['local_map']),)

        # Step 3: Global Map
        if self.config['global_map']:
            features += (self.get_map_from_state(fc_obs, hc_obs, problem, self.config['global_map']),)

        # Step 4: Meta Information
        if self.config['meta']:
            features += (self.get_meta_from_state(fc_obs, hc_obs, problem, self.config['meta']))

        return features if len(features) > 1 else features[0]

    def get_measurements_from_state(self, fc_obs, hc_obs, problem, config):
        self.measurements = np.append(self.measurements[1 - config:], np.array([[
            fc_obs.platform_state.lon.deg,
            fc_obs.platform_state.lat.deg,
            fc_obs.true_current_at_state.u,
            fc_obs.true_current_at_state.v
        ]]), axis=0)
        measurements = self.measurements - np.repeat(np.array([[1, 1, 0, 0]]), repeats=self.measurements.shape[0], axis=0) * self.measurements[-1]
        repeats = np.repeat(measurements[-1:], repeats=(config - self.measurements.shape[0]), axis=0)
        return np.append(measurements, repeats, axis=0)

    def get_map_from_state(self, fc_obs, hc_obs, problem, config):
        x_interval = [fc_obs.platform_state.lon.deg - config['xy_width_degree'] / 2, fc_obs.platform_state.lon.deg + config['xy_width_degree'] / 2]
        y_interval = [fc_obs.platform_state.lat.deg - config['xy_width_degree'] / 2, fc_obs.platform_state.lat.deg + config['xy_width_degree'] / 2]
        map = np.zeros((config['xy_width_points'], config['xy_width_points'], 0), dtype='float32')

        # Step 1: TTR Map on grid
        if config['features']['ttr_forecast']:
            ttr_forecast = self.forecast_planner.interpolate_value_function_in_hours(fc_obs, width_deg=config['xy_width_degree'], width=config['xy_width_points'])
            map = np.concatenate((map, np.expand_dims(ttr_forecast, axis=2)), axis=2, dtype='float32')
        if config['features']['ttr_hindcast']:
            ttr_hindcast = self.hindcast_planner.interpolate_value_function_in_hours(hc_obs, width_deg=config['xy_width_degree'], width=config['xy_width_points'])
            map = np.concatenate((map, np.expand_dims(ttr_hindcast, axis=2)), axis=2, dtype='float32')

        # Step 2: GP Observer on grid
        if len(config['features']['observer']) > 0:
            t_interval = [fc_obs.platform_state.date_time]
            self.observer.observe(fc_obs)
            self.observer.fit()
            gp_map = self.observer.get_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                t_interval=[t_interval[0], t_interval[-1]],
            ).interp(
                time=np.datetime64(fc_obs.platform_state.date_time.replace(tzinfo=None)),
                lon=np.linspace(x_interval[0], x_interval[1], config['xy_width_points']),
                lat=np.linspace(y_interval[0], y_interval[1], config['xy_width_points']),
                method='linear'
            )[config['features']['observer']['variables']].to_array().to_numpy().astype('float32').swapaxes(0,1).swapaxes(1,2)
            map = np.concatenate((map, gp_map), axis=2, dtype='float32')

        # Step 3: Hindcast Currents on grid
        if len(config['features']['currents_hindcast']) > 0:
            t_interval = [hc_obs.platform_state.date_time + datetime.timedelta(hours=h) for h in config['features']['currents_hindcast']]
            current_map = hc_obs.forecast_data_source.get_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                t_interval=[t_interval[0], t_interval[-1]],
            ).interp(
                time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                lon=np.linspace(x_interval[0], x_interval[1], config['xy_width_points']),
                lat=np.linspace(y_interval[0], y_interval[1], config['xy_width_points']),
                method='linear'
            )['water_u', 'water_v'].to_array().to_numpy().astype('float32').swapaxes(0,1).swapaxes(1,2)
            map = np.concatenate((map, current_map.squeeze()), axis=2, dtype='float32')

        # Step 3: Hindcast Currents on grid
        if len(config['features']['currents_forecast']) > 0:
            t_interval = [fc_obs.platform_state.date_time + datetime.timedelta(hours=h) for h in config['features']['currents_forecast']]
            current_map = fc_obs.forecast_data_source.get_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                t_interval=[t_interval[0], t_interval[-1]],
            ).interp(
                time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                lon=np.linspace(x_interval[0], x_interval[1], config['xy_width_points']),
                lat=np.linspace(y_interval[0], y_interval[1], config['xy_width_points']),
                method='linear'
            )['water_u', 'water_v'].to_array().to_numpy().astype('float32').swapaxes(0,1).swapaxes(1,2)
            map = np.concatenate((map, current_map.squeeze()), axis=2, dtype='float32')

        return map.squeeze()

    def get_meta_from_state(self, fc_obs, hc_obs, problem, config):
        features = []

        if 'lon' in config:
            features += fc_obs.platform_state.lon.deg
        if 'lat' in config:
            features += fc_obs.platform_state.lon.deg
        if 'time' in config:
            features += fc_obs.platform_state.date_time.timestamp()
        if 'target_distance' in config:
            features += problem.distance(fc_obs.platform_state)
        if 'target_direction' in config:
            features += problem.angle(fc_obs.platform_state)

        return features