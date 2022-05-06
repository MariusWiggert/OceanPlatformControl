import math
from collections import Callable

import numpy as np
from ray.rllib.agents.ppo import ppo

from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformAction
from ocean_navigation_simulator.env.controllers.Controller import Controller


class RLControllerFromAgent(Controller):
    """
    RL-based Controller using
    """

    def __init__(
        self,
        problem: Problem,
        agent,
        feature_constructor: Callable,
    ):
        """
        StraightLineController constructor
        Args:
            problem: the Problem the controller will run on
        """
        super().__init__(problem)

        self.agent = agent
        self.feature_constructor = feature_constructor

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """
        Return action that goes in the direction of the target with full power.
        Args:
            observation: observation returned by the simulator
        Returns:
            SimulatorAction dataclass
        """
        obs = self.feature_constructor(observation, self.problem)
        action = self.agent.compute_action(obs)

        # go towards the center of the target with full power
        return PlatformAction(magnitude=1, direction=action[0])