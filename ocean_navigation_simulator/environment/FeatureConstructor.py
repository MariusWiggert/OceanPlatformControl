import abc
import gym
import numpy as np

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Problem import Problem

"""
Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
and then convert to a numpy array that the RL model can use.
"""

class  FeatureConstructor(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_observation_space(config) -> gym.spaces.Box:
        pass

    @abc.abstractmethod
    def get_features_from_state(
        self,
        forecast_observation: ArenaObservation,
        hindcast_observation: ArenaObservation,
        problem: Problem
    ) -> np.ndarray:
        """
        Converts the observation to use relative positions
        Args:
            observation: current platform observation
            problem: class containing information about RL problem (end region, start state, etc.)
        Returns:
            numpy array containing relative lat pos, relative lon pos, elapsed time, u_curr, v_curr
        """
        pass