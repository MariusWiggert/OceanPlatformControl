from typing import Optional

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.RewardFunction import RewardFunction


class  OceanRewardFunction(RewardFunction):
    def __init__(self, planner: HJReach2DPlanner, config = {}, verbose: Optional[bool] = False):
        self.planner = planner
        self.verbose =verbose

    @staticmethod
    def get_reward_range():
        # Reward Upper Bound: at most 10min = 1/6h
        return (-1/6, 1/6)

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
            problem: class containing information about RL problem (end region, start state, etc.)
            prev_state: state the platform was at in the previous timestep
            curr_state: state the platform is at after taking the current action
            status:

        Returns:
            a float representing reward
        """
        prev_ttr = self.planner.interpolate_value_function_in_hours_at_point(observation=prev_obs)
        curr_ttr = self.planner.interpolate_value_function_in_hours_at_point(observation=curr_obs)

        return prev_ttr - curr_ttr