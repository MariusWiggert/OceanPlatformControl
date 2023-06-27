from typing import Union
import numpy as np
import jax.numpy as jnp


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