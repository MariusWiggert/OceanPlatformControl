# Define a navigation problem to be solved

from typing import Union
import numpy as np


# reward function: Option 1: just negative distance to target at each point in time
class RewardFunction:
    # x_domain = [-0, 2]
    # y_domain = [0, 1]

    def __init__(self, target: np.array, target_radius: float = 0.05):
        self.target = target
        self.target_radius = target_radius

    def get_reward(self, states: np.array) -> np.array:
        """Reward function for the navigation problem.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            rewards: vector of rewards as np.array (n,)
        """
        # return the negative distance
        rewards = -1.0 * self.get_distance_to_target(states)
        # rewards -= np.where(self.is_boundary(states), 100000.0, 0.0)
        return rewards

    def is_boundary(self, states: np.array) -> Union[float, np.array]:
        """Helper function to check if a state is in the boundary."""
        lon = states[:, 0]
        lat = states[:, 1]
        x_boundary = np.logical_or(
            lon < self.x_domain[0],
            lon > self.x_domain[1],
        )
        y_boundary = np.logical_or(
            lat < self.y_domain[0],
            lat > self.y_domain[1],
        )

        return np.logical_or(x_boundary, y_boundary)

    def reached_goal(self, states: np.array) -> Union[float, np.array]:
        """Helper function to check if a state reached the goal."""
        return self.get_distance_to_target(states) < 0.0

    def get_distance_to_target(self, states: np.array) -> Union[float, np.array]:
        """Helper function to get distance to target."""
        return np.linalg.norm(states[:, :2] - self.target, axis=1) - self.target_radius

    def check_terminal(self, states: np.array) -> np.array:
        """Check terminal conditions for the navigation problem.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            is_terminal: vector of boolean as np.array (n,)
        """
        # return np.logical_or(self.is_boundary(states), self.reached_goal(states))
        return self.reached_goal(states)

# Option 2: -1 when outside target and +100 when inside
class TimeRewardFunction(RewardFunction):
    def get_reward(self, states: np.array) -> np.array:
        """Reward function for the navigation problem.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            rewards: vector of rewards as np.array (n,)
        """
        # return reaching goal or terminal
        rewards = np.where(self.reached_goal(states), 0.0, -1.0)
        # rewards -= np.where(self.is_boundary(states), 100000.0, 0.0)
        # rewards -= 1.0
        return rewards