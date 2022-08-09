import abc
from typing import Tuple

from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState


class  RewardFunction(abc.ABC):
    @abc.abstractmethod
    def get_reward_range(self) -> Tuple:
        pass

    @abc.abstractmethod
    def get_reward(
        self,
        prev_state: PlatformState,
        curr_state: PlatformState,
        problem: NavigationProblem,
        problem_status: int
    ) -> float:
        pass