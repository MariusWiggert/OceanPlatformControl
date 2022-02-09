import numpy as np
from ocean_navigation_simulator.planners import HJReach2DPlanner
import hj_reachability as hj
import bisect


def check_feasibility2D(problem, T_hours_forward=100, deg_around_xt_xT_box=10, progress_bar=False,
                        stop_at_x_init=True, conv_m_to_deg=111120):
    """A function to run 2D time-optimal reachability and return the earliest arrival time in the x_T circle.
    returns feasibility (bool), T_earliest_in_h (float or None), feasibility_planner
    """
    # Specific settings to check feasibility
    specific_settings_dict = {
        'direction': 'forward',
        'T_goal_in_h': T_hours_forward,
        'initial_set_radii': [0.05, 0.05],
        'n_time_vector': 100,
        'grid_res': [0.04, 0.04],
        'deg_around_xt_xT_box': deg_around_xt_xT_box,
        'accuracy': 'high',
        'artificial_dissipation_scheme': 'local_local'}

    # Step 1: set up and run forward 2D Reachability
    feasibility_planner = HJReach2DPlanner(problem,
                                                specific_settings=specific_settings_dict,
                                                conv_m_to_deg=conv_m_to_deg)

    # load in the ground truth data
    feasibility_planner.update_forecast_dicts(problem.hindcasts_dicts)
    feasibility_planner.update_current_data(np.array(problem.x_0))

    # Step 2: run the hj planner
    x_0_rel = np.copy(problem.x_0)
    x_0_rel[3] = x_0_rel[3] - feasibility_planner.current_data_t_0

    # set the time_scales and offset in the non_dim_dynamics in which the PDE is solved
    feasibility_planner.nondim_dynamics.tau_c = feasibility_planner.specific_settings['T_goal_in_h'] * 3600
    feasibility_planner.nondim_dynamics.t_0 = x_0_rel[3]

    # set up the non_dimensional time-vector for which to save the value function
    solve_times = np.linspace(0, 1, feasibility_planner.specific_settings['n_time_vector'] + 1)
    feasibility_planner.nondim_dynamics.dimensional_dynamics.control_mode = 'max'

    # set variables to stop when x_end is in the reachable set
    if stop_at_x_init:
        stop_at_x_init = feasibility_planner.get_non_dim_state(
            feasibility_planner.get_x_from_full_state(
                feasibility_planner.x_T)
        )
    else:
        stop_at_x_init = None

    # create solver settings object
    solver_settings = hj.SolverSettings.with_accuracy(
        accuracy=feasibility_planner.specific_settings['accuracy'],
        x_init=stop_at_x_init,
        artificial_dissipation_scheme=feasibility_planner.diss_scheme)

    # solve the PDE in non_dimensional to get the value function V(s,t)
    non_dim_reach_times, feasibility_planner.all_values = hj.solve(
        solver_settings=solver_settings,
        dynamics=feasibility_planner.nondim_dynamics,
        grid=feasibility_planner.nonDimGrid,
        times=solve_times,
        initial_values=feasibility_planner.get_initial_values(center=x_0_rel, direction="forward"),
        progress_bar=progress_bar
    )

    # scale up the reach_times to be dimensional_times in seconds again
    feasibility_planner.reach_times = non_dim_reach_times * feasibility_planner.nondim_dynamics.tau_c + feasibility_planner.nondim_dynamics.t_0 + feasibility_planner.current_data_t_0

    reached, T_earliest_in_h = feasibility_planner.get_t_earliest_for_target_region()
    # not reached
    if reached == False:
        print("Not_reached")
        return False, None, feasibility_planner
    else:
        print("reached earliest after h: ", T_earliest_in_h)
        # extract time-optimal trajectory from the run
        feasibility_planner.extract_trajectory(
            x_start=feasibility_planner.get_x_from_full_state(feasibility_planner.x_T),
            traj_rel_times_vector=None)
        return True, T_earliest_in_h, feasibility_planner


def check_if_x_is_reachable(feasibility_planner, x, time):
    """ Function to check if a certain point x is reachable at the datetime time using a run planner object."""
    # check if the planner was run already
    if feasibility_planner.all_values is None:
        raise ValueError("the reachability planner needs to be run before we can check reachability.")
    times = feasibility_planner.reach_times
    if times[0] < times[-1]:  # ascending times
        idx_start_time_closest = bisect.bisect_right(times, time.timestamp(), hi=len(times) - 1)
    else:  # descending times
        idx_start_time_closest = -bisect.bisect_right(times[::-1], time.timestamp())

    return feasibility_planner.grid.interpolate(feasibility_planner.all_values[idx_start_time_closest, ...], x) <= 0