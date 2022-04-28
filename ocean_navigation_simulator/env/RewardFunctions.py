from ocean_navigation_simulator.env import utils
from ocean_navigation_simulator.env.PlatformState import PlatformState


def double_gyre_reward_function(prev_state: PlatformState, curr_state: PlatformState, target: PlatformState,
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
    # TODO: should reward include time elapsed per timestep?

    bonus = 200
    prev_distance = utils.euclidean_distance(prev_state, target)
    curr_distance = utils.euclidean_distance(curr_state, target)

    if done:
        return prev_distance - curr_distance + bonus
    else:
        return prev_distance - curr_distance
