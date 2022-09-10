import abc
from typing import Tuple

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState


class  RewardFunction(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_reward_range() -> Tuple:
        pass

    @abc.abstractmethod
    def get_reward(
        self,
        prev_state: ArenaObservation,
        curr_state: ArenaObservation,
        problem: NavigationProblem,
        problem_status: int
    ) -> float:
        pass