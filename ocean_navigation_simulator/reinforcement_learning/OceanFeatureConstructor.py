import datetime
import math
import gym
import numpy as np

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.FeatureConstructor import FeatureConstructor
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem

"""
Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
and then convert to a numpy array that the RL model can use.
"""
class  OceanFeatureConstructor(FeatureConstructor):
    def __init__(self, config = {}):
        self.config = {
            'num_measurements': 5,
            'forecast_map_width_degree': 1,
            'forecast_map_width_points': 50,
            'forecast_time_width_hours': 120,
            'forecast_time_points': 10,
        } | config
        self.measurements = np.zeros(shape=(self.config['num_measurements'], 4))

    def get_observation_space(self):
        number_of_features = self.config['num_measurements'] * 4 + self.config['forecast_map_width_points']*(self.config['forecast_map_width_points']**2)

        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(number_of_features,)
        )

    def get_features_from_state(self, obs: ArenaObservation, problem: NavigationProblem) -> np.ndarray:
        # Step 1: Save Measurements
        self.measurements = np.append(self.measurements, np.expand_dims(np.array([
            obs.platform_state.lon.deg,
            obs.platform_state.lat.deg,
            obs.true_current_at_state.u,
            obs.true_current_at_state.v
        ]), axis=0), axis=0)
        self.measurements = self.measurements[-self.config['num_measurements']:]

        # Step 2: Calculate relative measurements
        measurements = self.measurements
        measurements[:, :2] = self.measurements[:, :2] - np.ones((self.config['num_measurements'], 2)) * self.measurements[-1, :2] * 2
        print(measurements)

        # Step 3: Get forecast
        forecast = obs.forecast_data_source.get_data_over_area(
            x_interval=[obs.platform_state.lon.deg-self.config['forecast_map_width_degree']/2,obs.platform_state.lon.deg+self.config['forecast_map_width_degree']/2],
            y_interval=[obs.platform_state.lat.deg-self.config['forecast_map_width_degree']/2,obs.platform_state.lat.deg+self.config['forecast_map_width_degree']/2],
            t_interval=[obs.platform_state.date_time, obs.platform_state.date_time+datetime.timedelta(hours=self.config['forecast_time_width_hours'])],
            spatial_resolution=self.config['forecast_map_width_degree']/(self.config['forecast_map_width_points']-2),
            temporal_resolution=int(3600*self.config['forecast_time_width_hours']/(self.config['forecast_time_points']-2)),
        )
        print(forecast)
        forecast = forecast.to_array().to_numpy()
        print(forecast.shape)
        return np.concatenate((forecast.flatten(), measurements.flatten()))
