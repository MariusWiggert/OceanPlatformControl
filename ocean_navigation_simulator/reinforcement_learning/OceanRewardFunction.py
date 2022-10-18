import math
from typing import Optional

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.reinforcement_learning.RewardFunction import (
    RewardFunction,
)


class OceanRewardFunction(RewardFunction):
    def __init__(
        self,
        forecast_planner: HJReach2DPlanner,
        hindcast_planner: HJReach2DPlanner,
        config={},
    ):
        self.forecast_planner = forecast_planner
        self.hindcast_planner = hindcast_planner
        self.config = config

    @staticmethod
    def get_reward_range(config={}):
        # Reward Upper Bound: at most 10min = 1/6h
        return -float("inf"), float("inf")

    def get_reward(
        self,
        prev_fc_obs: ArenaObservation,
        curr_fc_obs: ArenaObservation,
        prev_hc_obs: ArenaObservation,
        curr_hc_obs: ArenaObservation,
        problem: NavigationProblem,
        problem_status: int,
    ) -> float:
        """
        Flexible Reward function which can be customized by config.
        Args:
                :param prev_obs: state the platform was at in the previous timestep
                :param curr_obs: state the platform is at after taking the current action
                :param problem: class containing information about RL problem (end region, start state, etc.)
                :param problem_status:

        Returns:
                a float representing reward
        """
        reward = 0

        if problem_status > 0 and self.config["target_bonus"] > 0:
            reward += self.config["target_bonus"]

        if problem_status == 0:
            if self.config["delta_ttr_forecast"] > 0:
                prev_ttr_fc = self.forecast_planner.interpolate_value_function_in_hours(
                    observation=prev_fc_obs
                ).item()
                curr_ttr_fc = self.forecast_planner.interpolate_value_function_in_hours(
                    observation=curr_fc_obs
                ).item()
                reward += self.config["delta_ttr_forecast"] * (prev_ttr_fc - curr_ttr_fc)
            if self.config["delta_ttr_hindcast"] > 0:
                prev_ttr_hc = self.hindcast_planner.interpolate_value_function_in_hours(
                    observation=prev_hc_obs
                ).item()
                curr_ttr_hc = self.hindcast_planner.interpolate_value_function_in_hours(
                    observation=curr_hc_obs
                ).item()
                reward += self.config["delta_ttr_hindcast"] * (prev_ttr_hc - curr_ttr_hc)

        if self.config["step_punishment"] > 0:
            reward -= self.config["step_punishment"]

        if problem_status > 0 and self.config["fail_punishment"] > 0:
            reward -= self.config["fail_punishment"]

        return reward
