# TODO: this is not working after refactoring yet!
import numpy as np
from ocean_navigation_simulator.planners import HJReach2DPlanner
import ocean_navigation_simulator
import hj_reachability as hj
import bisect
from scipy.interpolate import interp1d
import jax.numpy as jnp
from functools import partial
import math


def check_feasibility_2D_w_sim(
    problem,
    T_hours_forward,
    deg_around_xt_xT_box,
    grid_res=None,
    conv_m_to_deg=111120,
    hours_to_hj_solve_timescale=3600,
    sim_dt=600,
    sim_deg_around_x_t=3,
):
    """Function to calculate the earliest time-to-reach by running multi-reachability on the true currents and running
    the simulator using multi-reachability closed-loop with small temporal steps. This doesn't overestimate."""

    # Step 0: Set problem to plan and execute on true currents
    plan_setting = problem.plan_on_gt
    if grid_res is None:
        grid_res = [0.04, 0.04]
    problem.plan_on_gt = True
    # Step 1: Set-Up the planner and simulator dicts
    sim_config_dict = {
        "dt": sim_dt,  # in seconds
        "conv_m_to_deg": conv_m_to_deg,  # used to transform anything in meters to in degrees (assuming a sphere)
        "int_pol_type": "linear",
        "sim_integration": "ef",
        "hours_to_sim_timescale": hours_to_hj_solve_timescale,
        "deg_around_x_t": sim_deg_around_x_t,  # degrees around x_0 that are read in when simulating
        "t_horizon_sim": math.ceil(
            T_hours_forward / 24
        ),  # time horizon for data sub-setting in days
    }
    control_config_dict = {
        "PlannerConfig": {
            "name": "HJReach2DPlanner",
            "dt_replanning": 3600000000000.0,
            "specific_settings": {
                "direction": "multi-reach-back",
                "d_max": 0.0,
                "T_goal_in_h": T_hours_forward,
                "hours_to_hj_solve_timescale": hours_to_hj_solve_timescale,
                "n_time_vector": 100,
                "grid_res": grid_res,
                "deg_around_xt_xT_box": deg_around_xt_xT_box,
                "accuracy": "high",
                "boundary_buffer": 0.05,
                "artificial_dissipation_scheme": "local_local",
                "progress_bar": False,
            },
        },
        "SteeringCtrlConfig": {"name": "None", "dt_replanning": 360000000000.0},
    }

    # Step 2: Run the simulator
    sim = ocean_navigation_simulator.simulator.OceanNavSimulator(
        sim_config_dict=sim_config_dict, control_config_dict=control_config_dict, problem=problem
    )

    # Step 3: Extract if it reached and if yes the time.
    sim.evaluation_run(T_in_h=T_hours_forward)

    # Step 4: Make problem object back to plan_on_gt as original
    problem.plan_on_gt = plan_setting

    if sim.termination_reason != "goal_reached":
        print("Not_reached")
        return False, None, sim
    else:
        T_earliest_in_h = (
            sim.trajectory[3, -1] - sim.trajectory[3, 0]
        ) / hours_to_hj_solve_timescale
        print("reached earliest after h: ", T_earliest_in_h)
        # return sim object to extract the time-optimal trajectory & ctrl sequence from
        return True, T_earliest_in_h, sim


def check_feasibility2D(
    problem,
    T_hours_forward,
    deg_around_xt_xT_box,
    progress_bar=False,
    conv_m_to_deg=111120,
    hours_to_hj_solve_timescale=3600,
    grid_res=[0.04, 0.04],
):
    """Function to calculate the earliest time to reach by running Multi-Time Reachability backwards from T_hours_forward
    in the future and getting the fastest-time-to-reach via the interpolation of the value function at t=0.

    Note: Because of numerical issues of gradient estimation near the target (flat inside, gradient outside) this
    currently leads to systematic overestimation of the earliest-to-reach by 4-8% (depending on the specifics).
    This can be reduced by decreasing the grid_resolution.
    It can be fixed long-term by (a) have the boundary be a grid point (but impossible >1D) or (b) an upwinding schema
    at the boundary of the flat target or a bit hacky (c) by running it without a flat bottom inside the target e.g.
    a target point (instead of a circle) or just lowering it and after solving the HJ PDE subtracting that from inside.
    """
    # Specific settings to check feasibility
    specific_settings_dict = {
        "direction": "multi-reach-back",
        "T_goal_in_h": T_hours_forward,
        "hours_to_hj_solve_timescale": hours_to_hj_solve_timescale,
        "n_time_vector": 100,
        "d_max": 0.0,
        "grid_res": grid_res,
        "deg_around_xt_xT_box": deg_around_xt_xT_box,
        "accuracy": "high",
        "artificial_dissipation_scheme": "local_local",
    }

    # Step 1: set up and run forward 2D Reachability
    feasibility_planner = HJReach2DPlanner(
        problem, specific_settings=specific_settings_dict, conv_m_to_deg=conv_m_to_deg
    )

    # load in the ground truth data
    feasibility_planner.update_forecast_dicts(problem.hindcast_data_source)
    feasibility_planner.update_current_data(np.array(problem.x_0))

    # Step 2: run the hj planner
    x_0_rel = np.copy(problem.x_0)
    x_0_rel[3] = x_0_rel[3] - feasibility_planner.current_data_t_0

    # set the time_scales and offset in the non_dim_dynamics in which the PDE is solved
    feasibility_planner.nondim_dynamics.tau_c = (
        feasibility_planner.specific_settings["T_goal_in_h"] * hours_to_hj_solve_timescale
    )
    feasibility_planner.nondim_dynamics.t_0 = x_0_rel[3]

    # set up the non_dimensional time-vector for which to save the value function
    solve_times = np.linspace(0, 1, feasibility_planner.specific_settings["n_time_vector"] + 1)
    solve_times = np.flip(solve_times, axis=0)
    feasibility_planner.nondim_dynamics.dimensional_dynamics.control_mode = "min"

    # write multi_reach hamiltonian postprocessor
    def multi_reach_step(mask, val):
        val = jnp.where(mask <= 0, -1, val)
        return val

    initial_values = feasibility_planner.get_initial_values(
        center=feasibility_planner.x_T, direction="multi-reach-back"
    )

    # create solver settings object
    solver_settings = hj.SolverSettings.with_accuracy(
        accuracy=feasibility_planner.specific_settings["accuracy"],
        artificial_dissipation_scheme=feasibility_planner.diss_scheme,
        hamiltonian_postprocessor=partial(multi_reach_step, initial_values),
    )

    # solve the PDE in non_dimensional to get the value function V(s,t)
    non_dim_reach_times, feasibility_planner.all_values = hj.solve(
        solver_settings=solver_settings,
        dynamics=feasibility_planner.nondim_dynamics,
        grid=feasibility_planner.nonDimGrid,
        times=solve_times,
        initial_values=initial_values,
        progress_bar=progress_bar,
    )

    # scale up the reach_times to be dimensional_times in seconds again
    feasibility_planner.reach_times = (
        non_dim_reach_times * feasibility_planner.nondim_dynamics.tau_c
        + feasibility_planner.nondim_dynamics.t_0
    )
    # in times log the posix time because we need it for check_if_x_is_reachable computation
    feasibility_planner.times = (
        feasibility_planner.reach_times + feasibility_planner.current_data_t_0
    )

    # Step 1: Check if the initial point is reachable at all
    non_dim_val_at_x_0 = feasibility_planner.grid.interpolate(
        feasibility_planner.all_values[-1, ...],
        feasibility_planner.get_x_from_full_state(feasibility_planner.x_0),
    )

    if non_dim_val_at_x_0 > 0:
        print("Not_reached")
        return False, None, feasibility_planner
    else:
        # Dimensionalize the non_dim_val_at_x_0 with T_hours_forward and add T_hours_forward to get reachable time
        T_earliest_in_h = non_dim_val_at_x_0 * T_hours_forward + T_hours_forward
        print("reached earliest after h: ", T_earliest_in_h)
        # extract time-optimal trajectory from the run

        def termination_condn(x_target, r, x, t):
            return np.linalg.norm(x_target - x) <= r

        feasibility_planner.extract_trajectory(
            x_start=feasibility_planner.get_x_from_full_state(x_0_rel.flatten()),
            termination_condn=partial(
                termination_condn, feasibility_planner.x_T, feasibility_planner.problem.x_T_radius
            ),
        )
        return True, T_earliest_in_h, feasibility_planner


def run_forward_reachability(
    problem,
    T_hours_forward=100,
    deg_around_xt_xT_box=10,
    progress_bar=False,
    stop_at_x_init=False,
    conv_m_to_deg=111120,
    initial_set_radii=[0.05, 0.05],
    grid_res=[0.04, 0.04],
    hours_to_hj_solve_timescale=3600,
):
    """A function to run 2D time-optimal reachability and return the earliest arrival time in the x_T circle.
    returns feasibility (bool), T_earliest_in_h (float or None), feasibility_planner
    """
    # Specific settings to check feasibility
    specific_settings_dict = {
        "direction": "forward",
        "T_goal_in_h": T_hours_forward,
        "initial_set_radii": initial_set_radii,
        "hours_to_hj_solve_timescale": hours_to_hj_solve_timescale,
        "n_time_vector": 500,
        "d_max": 0.0,
        "boundary_buffer": 0.05,
        "grid_res": grid_res,
        "deg_around_xt_xT_box": deg_around_xt_xT_box,
        "accuracy": "high",
        "artificial_dissipation_scheme": "local_local",
    }

    # Step 1: set up and run forward 2D Reachability
    feasibility_planner = HJReach2DPlanner(
        problem, specific_settings=specific_settings_dict, conv_m_to_deg=conv_m_to_deg
    )

    # load in the ground truth data
    feasibility_planner.update_forecast_dicts(problem.hindcast_data_source)
    feasibility_planner.update_current_data(np.array(problem.x_0))

    # Step 2: run the hj planner
    x_0_rel = np.copy(problem.x_0)
    x_0_rel[3] = x_0_rel[3] - feasibility_planner.current_data_t_0
    # set the x_t for the feasibility planner
    feasibility_planner.x_t = x_0_rel

    # set the time_scales and offset in the non_dim_dynamics in which the PDE is solved
    feasibility_planner.nondim_dynamics.tau_c = (
        feasibility_planner.specific_settings["T_goal_in_h"] * hours_to_hj_solve_timescale
    )
    feasibility_planner.nondim_dynamics.t_0 = x_0_rel[3]

    # set up the non_dimensional time-vector for which to save the value function
    solve_times = np.linspace(0, 1, feasibility_planner.specific_settings["n_time_vector"] + 1)
    feasibility_planner.nondim_dynamics.dimensional_dynamics.control_mode = "max"

    # set variables to stop when x_end is in the reachable set
    if stop_at_x_init:
        stop_at_x_init = feasibility_planner.get_non_dim_state(
            feasibility_planner.get_x_from_full_state(feasibility_planner.x_T)
        )
    else:
        stop_at_x_init = None

    # create solver settings object
    solver_settings = hj.SolverSettings.with_accuracy(
        accuracy=feasibility_planner.specific_settings["accuracy"],
        x_init=stop_at_x_init,
        artificial_dissipation_scheme=feasibility_planner.diss_scheme,
    )

    # solve the PDE in non_dimensional to get the value function V(s,t)
    non_dim_reach_times, feasibility_planner.all_values = hj.solve(
        solver_settings=solver_settings,
        dynamics=feasibility_planner.nondim_dynamics,
        grid=feasibility_planner.nonDimGrid,
        times=solve_times,
        initial_values=feasibility_planner.get_initial_values(direction="forward"),
        progress_bar=progress_bar,
    )

    # scale up the reach_times to be dimensional_times in seconds again
    feasibility_planner.reach_times = (
        non_dim_reach_times * feasibility_planner.nondim_dynamics.tau_c
        + feasibility_planner.nondim_dynamics.t_0
    )
    # in times log the posix time because we need it for check_if_x_is_reachable computation
    feasibility_planner.times = (
        feasibility_planner.reach_times + feasibility_planner.current_data_t_0
    )

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
            traj_rel_times_vector=None,
        )
        return True, T_earliest_in_h, feasibility_planner
