from typing import Union
import numpy as np
import jax.numpy as jnp


# Note: this only works for accurate planning =)
def get_value_from_hj(belief, hj_planner):
    """
    :param belief: belief with states and weights
    :param hj_planner: hj_reachability planner object
    :return: time-to-reach value at the given belief
    """
    belief_states = belief.states
    rel_time_in_seconds = np.zeros_like(belief_states[0, 2])
    # rel_time_in_seconds = belief_states[0, 2]

    time = rel_time_in_seconds + hj_planner.reach_times[0]
    state = jnp.array([belief_states[0, 0], belief_states[0, 1]])
    non_dim_value = hj_planner.grid.spacetime_interpolate(hj_planner.reach_times, hj_planner.all_values, time, state)
    # % now transform to time-to-reach value
    total_time_in_s = jnp.abs(rel_time_in_seconds - hj_planner.reach_times[-1])
    dim_value = non_dim_value * total_time_in_s
    # now to time-to-reach
    TTR_value = total_time_in_s + dim_value - rel_time_in_seconds
    # now upscale to same time-resolution as the reward by 10 and add 100 as final end reward.
    reward_value = -(TTR_value * 10) # + 100
    return reward_value.__float__()

def get_value_from_hj_dict(belief, hj_planner_dict):
    """
    :param belief: belief with states and weights
    :param hj_planner: hj_reachability planner object
    :return: time-to-reach value at the given belief
    """
    belief_states = belief.states
    rel_time_in_seconds = np.zeros_like(belief_states[0, 2])
    # rel_time_in_seconds = belief_states[0, 2]

    # select which hj_planner to use
    value_estimates = []
    for highway_value in belief_states[:, 3]:
        if highway_value == 0:
            hj_planner = hj_planner_dict['zero']
        elif highway_value > 0:
            hj_planner = hj_planner_dict['true']
        elif highway_value < 0:
            hj_planner = hj_planner_dict['minus']

        time = rel_time_in_seconds + hj_planner.reach_times[0]
        state = jnp.array([belief_states[0, 0], belief_states[0, 1]])

        non_dim_value = hj_planner.grid.spacetime_interpolate(hj_planner.reach_times, hj_planner.all_values, time, state)
        # % now transform to time-to-reach value
        total_time_in_s = jnp.abs(rel_time_in_seconds - hj_planner.reach_times[-1])
        dim_value = non_dim_value * total_time_in_s
        # now to time-to-reach
        TTR_value = total_time_in_s + dim_value - rel_time_in_seconds
        # now upscale to same time-resolution as the reward by 10 and add 100 as final end reward.
        reward_value = -(TTR_value * 10)  # + 100
        value_estimates.append(reward_value.__float__())

    # now weight and calculate
    value_estimates = np.array(value_estimates)
    value_estimates = (value_estimates * belief.weights).sum()
    return value_estimates
