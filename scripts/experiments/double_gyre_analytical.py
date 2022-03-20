from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import OceanNavSimulator
from ocean_navigation_simulator import utils
import numpy as np
import time

#% Settings to feed into the planner
# Set the platform configurations
platform_config_dict = {'battery_cap': 400.0, 'u_max': 0.1, 'motor_efficiency': 1.0,
                        'solar_panel_size': 0.5, 'solar_efficiency': 0.2, 'drag_factor': 675}


hindcast_field = utils.analytical_fields.PeriodicDoubleGyre(
    spatial_output_shape=(50,50),
    temporal_domain=[-10, 1000],
    temporal_default_length=200,
    v_amplitude=0.2,
    epsilon_sep=0.15,
    period_time=5)

forecast_field = utils.analytical_fields.PeriodicDoubleGyre(
    spatial_output_shape=(50,50),
    temporal_domain=[-10, 1000],
    temporal_default_length=200,
    v_amplitude=0.1,
    epsilon_sep=0.15,
    period_time=5)

#%
t_0 = 0.
x_0 = [0.25, 0.25, 1]  # lon, lat, battery
x_T = [1.75, 0.75]
hindcast_source = {'data_source_type': 'analytical_function',
                   'data_source': hindcast_field}

forecasts_source = {'data_source_type': 'analytical_function',
                   'data_source': forecast_field}

plan_on_gt=False
prob = Problem(x_0, x_T, t_0,
               platform_config_dict=platform_config_dict,
               hindcast_source= hindcast_source,
               forecast_source=forecasts_source,
               plan_on_gt = plan_on_gt,
               x_T_radius=0.1)
#%%
prob.viz(cut_out_in_deg=10) # plots the current at t_0 with the start and goal position
# # create a video of the underlying currents rendered in Jupyter, Safari or as file
# prob.viz(video=True) # renders in Jupyter
# prob.viz(video=True, html_render='safari',
#          temp_horizon_viz_in_h=30,
#          hours_to_hj_solve_timescale=1)    # renders in Safari
# prob.viz(video=True, filename='problem_animation.gif', temp_horizon_viz_in_h=200) # saves as gif file
#%% Check feasibility of the problem
feasible, earliest_T, gt_run_simulator = utils.check_feasibility_2D_w_sim(
    problem=prob, T_hours_forward=10, deg_around_xt_xT_box=2, sim_dt=0.01,
    conv_m_to_deg=1, grid_res=[0.02, 0.02], hours_to_hj_solve_timescale=1)
# 5.89999. With smaller dt we get 5.809999999999921 => which is below the HJ result.
#%% Plot the time-optimal trajectory (if true currents are known)
gt_run_simulator.plot_trajectory(plotting_type='2D')
gt_run_simulator.plot_trajectory(plotting_type='ctrl')
gt_run_simulator.plot_trajectory(plotting_type='2D_w_currents_w_controls')
#%% Visualize the Multi-Time Reachability Level sets
gt_run_simulator.high_level_planner.plot_reachability_snapshot(
    rel_time_in_h=0, multi_reachability=True, granularity_in_h=1., time_to_reach=False)
# gt_run_simulator.high_level_planner.plot_reachability_animation(
#     type='mp4', multi_reachability=True, granularity_in_h=1, time_to_reach=True)
# #%% Older version of calculating earliest-to-reach (faster but overestimates it systematically)
# feasible, earliest_T, feasibility_planner = utils.feasibility_check.check_feasibility2D(
#     problem=prob, T_hours_forward=10, deg_around_xt_xT_box=2,
#     conv_m_to_deg=1, grid_res=[0.02, 0.02], hours_to_hj_solve_timescale=1)
# #%% Plot best-in-hindsight trajectory
# feasibility_planner.plot_2D_traj()
# feasibility_planner.plot_ctrl_seq()
# #%%
# feasibility_planner.plot_reachability_animation(type='mp4', multi_reachability=True, granularity_in_h=0.2, time_to_reach=True)
# # feasibility_planner.plot_reachability_snapshot(rel_time_in_h=-14,
# #                                                multi_reachability=True,
# #                                                granularity_in_h=1.,
# #                                                time_to_reach=False)
#%% Set-Up and run Simulator
sim = OceanNavSimulator(sim_config_dict="simulator_analytical.yaml",
                        control_config_dict='reach_controller.yaml',
                        problem=prob)
start = time.time()
sim.run(T_in_h=12)
print("Took : ", time.time() - start)
print("Arrived after {} h".format(sim.trajectory[3,-1] - sim.trajectory[3,0]))
# Multi-reach arrives after 17.25h (should be impossible as that's better than in hindsight...)
# Backwards with T_goal_in_h=17.25h arrives after
# => which one is correct?
#%% Step 5: plot results from Simulator
# # plot Battery levels over time
sim.plot_trajectory(plotting_type='battery')
# # # plot 2D Trajectory without background currents
sim.plot_trajectory(plotting_type='2D')
# # plot control over time
sim.plot_trajectory(plotting_type='ctrl')
# # plot 2D Trajectory with currents at t_0
sim.plot_trajectory(plotting_type='2D_w_currents')
# # plot 2D Trajectory with currents at t_0 and control_vec for each point
sim.plot_trajectory(plotting_type='2D_w_currents_w_controls')
# plot with plans
# sim.plot_2D_traj_w_plans(plan_idx=None, with_control=False, underlying_plot_type='2D', plan_temporal_stride=1)
#%% Step 5:2 create animations of the simulation run
sim.create_animation(vid_file_name="simulation_animation.mp4", deg_around_x0_xT_box=0.5,
                     temporal_stride=1, time_interval_between_pics=200,
                     linewidth=1.5, marker='x', linestyle='--', add_ax_func=None)
#%% Test how I can add the best-in-hindsight trajectory to it
from functools import partial
x_best = np.linspace(x_0[0],x_T[0],20).reshape(1,-1)
y_best = np.linspace(x_0[1],x_T[1],20).reshape(1,-1)
x_traj_best = np.vstack((x_best, y_best))

def plot_best(ax, time, x_traj_best):
    ax.plot(x_traj_best[0,:], x_traj_best[1,:], color='k', label="best in hindsight")

to_add = partial(plot_best, x_traj_best=x_traj_best)
sim.create_animation_with_plans(vid_file_name="simulation_animation_with_plans.mp4",
                                plan_temporal_stride=5, alpha=1, with_control=True,
                                plan_traj_color='sienna', plan_u_color='blue', add_ax_func=to_add)
#%% Plot the latest Plan from reachability planner
sim.high_level_planner.plot_2D_traj()
sim.high_level_planner.plot_ctrl_seq()
#%%
sim.high_level_planner.plot_reachability_snapshot(rel_time_in_h=0,
                                               multi_reachability=True,
                                               granularity_in_h=1.,
                                               time_to_reach=False)
#%% Plot 2D reachable set evolution
sim.high_level_planner.plot_reachability_animation(type='mp4', multi_reachability=True, granularity_in_h=1, time_to_reach=True)