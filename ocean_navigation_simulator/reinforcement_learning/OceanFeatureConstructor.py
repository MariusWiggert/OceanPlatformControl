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
    def __init__(self, config):
        self.config = {
            'measurements': 5,
            'degree_around': 1,
        } | config
        self.measurements = np.zeros(shape=(0, 4))

    def get_observation_space(self):
        number_of_features = self.config['measurements'] * 4 +
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(number_of_features,)
        )

    def get_features_from_state(self, obs: ArenaObservation, problem: NavigationProblem) -> np.ndarray:
        measurement = np.array([
            obs.platform_state.lon.deg,
            obs.platform_state.lat.deg,
            obs.true_current_at_state.u,
            obs.true_current_at_state.v
        ])
        np.append(self.measurements, np.expand_dims(measurement.squeeze(), axis=0), axis=0)
        self.measurements = self.measurements[-self.config['']:]

        forecast = obs.forecast_data_source.get_data_over_area(
            x_interval=[],
            y_interval=[],
            t_interval=List[datetime.datetime]
        )

        return np.concatenate((forecast.flatten(), self.measurements.flatten()))
