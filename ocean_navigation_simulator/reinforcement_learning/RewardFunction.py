import abc
from typing import Tuple

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)


class RewardFunction(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_reward_range() -> Tuple:
        pass

    @abc.abstractmethod
    def get_reward(
        self,
        prev_obs: ArenaObservation,
        curr_obs: ArenaObservation,
        problem: NavigationProblem,
        problem_status: int,
    ) -> float:
        pass
