import numpy as np

from DoubleGyreProblem import DoubleGyreProblem
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint


def euclidean_distance(state, target):
    return np.sqrt((state.lat.deg - target.lat.deg) ** 2 + (state.lon.deg - target.lon.deg) ** 2)


def double_gyre_reward_function(prev_state: PlatformState, curr_state: PlatformState, problem: DoubleGyreProblem,
                                done: bool) -> float:
    """
    Reward function based on double gyre paper
    Args:
        problem: class containing information about RL problem (end region, start state, etc.)
        prev_state: state the platform was at in the previous timestep
        curr_state: state the platform is at after taking the current action
        done: if simulation episode, True, otherwise, False

    Returns:
        a float representing reward
    """
    target = problem.end_region
    bonus = 200  # TODO: change to make right amount
    prev_distance = euclidean_distance(prev_state, target)
    curr_distance = euclidean_distance(curr_state, target)

    time_diff = (curr_state.date_time - prev_state.date_time).total_seconds()

    if done:
        return prev_distance - curr_distance - time_diff + bonus
    else:
        return prev_distance - curr_distance - time_diff
