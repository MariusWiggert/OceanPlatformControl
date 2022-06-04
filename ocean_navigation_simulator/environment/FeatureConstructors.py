import abc

import gym
import numpy as np

from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.NavigationProblem import NavigationProblem

"""
Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
and then convert to a numpy array that the RL model can use.
"""

class  FeatureConstructor:
    @abc.abstractmethod
    def get_observation_space(self) -> gym.spaces.Box:
        pass

    @abc.abstractmethod
    def get_features_from_state(self, obs: ArenaObservation, problem: NavigationProblem) -> np.ndarray:
        pass