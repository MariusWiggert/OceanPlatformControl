from ocean_navigation_simulator.env import utils
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint


def double_gyre_reward_function(prev_state: PlatformState, curr_state: PlatformState, target: SpatialPoint,
                                done: bool) -> float:
    """
    Reward function based on double gyre paper
    Args:
        prev_state: state the platform was at in the previous timestep
        curr_state: state the platform is at after taking the current action
        target: goal end state
        done: if simulation episode, True, otherwise, False

    Returns:
        a float representing reward
    """
    bonus = 200  # TODO: change to make right amount
    prev_distance = utils.euclidean_distance(prev_state, target)
    curr_distance = utils.euclidean_distance(curr_state, target)

    time_diff = (curr_state.date_time - prev_state.date_time).total_seconds()

    if done:
        return prev_distance - curr_distance - time_diff + bonus
    else:
        return prev_distance - curr_distance - time_diff
