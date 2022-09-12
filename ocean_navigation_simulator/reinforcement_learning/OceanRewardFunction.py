from typing import Optional

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.reinforcement_learning.RewardFunction import RewardFunction


class  OceanRewardFunction(RewardFunction):
    def __init__(self, planner: HJReach2DPlanner, config = {}, verbose: Optional[int] = 0):
        self.planner = planner
        self.config = config
        self.verbose =verbose

    @staticmethod
    def get_reward_range(config = {}):
        # Reward Upper Bound: at most 10min = 1/6h
        return -float("inf"), float("inf")

    def get_reward(
        self,
        prev_obs: ArenaObservation,
        curr_obs: ArenaObservation,
        problem: NavigationProblem,
        problem_status: int
    ) -> float:
        """
        Reward function based on double gyre paper
        Args:
            :param prev_obs: state the platform was at in the previous timestep
            :param curr_obs: state the platform is at after taking the current action
            :param problem: class containing information about RL problem (end region, start state, etc.)
            :param problem_status:

        Returns:
            a float representing reward
        """
        prev_ttr = self.planner.interpolate_value_function_in_hours(observation=prev_obs).item()
        curr_ttr = self.planner.interpolate_value_function_in_hours(observation=curr_obs).item()

        return (prev_ttr - curr_ttr) + (self.config['target_bonus'] if problem_status > 0 else 0) + (self.config['fail_punishment'] if problem_status < 0 else 0)