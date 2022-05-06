from ocean_navigation_simulator.env.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.env.PlatformState import PlatformState


def double_gyre_reward_function(
    prev_state: PlatformState,
    curr_state: PlatformState,
    problem: NavigationProblem,
    solved: bool,
    crashed: bool
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
    bonus = 200
    penalty = -200

    prev_distance = prev_state.distance(problem.end_region)
    curr_distance = curr_state.distance(problem.end_region)
    distance_improvement = prev_distance - curr_distance

    time_diff = (curr_state.date_time - prev_state.date_time).total_seconds()

    return - time_diff + 100 * distance_improvement + (bonus if solved else 0) + (penalty if crashed else 0)