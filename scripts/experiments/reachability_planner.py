import sys
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import OceanNavSimulator
from ocean_navigation_simulator import utils
from ocean_navigation_simulator.utils.analytical_fields import FixedCurrentHighwayField
import numpy as np
import jax.numpy as jnp
import datetime
import matplotlib.pyplot as plt
import os
import hj_reachability as hj
import time

#% Settings to feed into the planner
# Set the platform configurations
platform_config_dict = {'battery_cap': 400.0, 'u_max': 0.1, 'motor_efficiency': 1.0,
                        'solar_panel_size': 0.5, 'solar_efficiency': 0.2, 'drag_factor': 675}

# Create the navigation problem
t_0 = datetime.datetime(2021, 11, 23, 12, 10, 10, tzinfo=datetime.timezone.utc)
# all through Gulf of Mexico
# x_0 = [-94.5, 25, 1]  # lon, lat, battery
# x_T = [-83, 25]
# # # short 80h start-goal Mission
# x_0 = [-88.25, 26.5, 1]  # lon, lat, battery
# x_T = [-90, 26]
# For head figure bifurcating flow
x_0 = [-88.25, 26.5, 1]  # lon, lat, battery
x_T = [-87, 26]
# With eddie below (after 100h)
t_0 = datetime.datetime(2021, 11, 23, 12, 10, 10, tzinfo=datetime.timezone.utc)
x_0 = [-81.5, 23.5, 1]  # lon, lat, battery
x_T = [-80.4, 24.2]
# x_T = [-79.8, 24.]
# hindcast_folder = "data/single_day_hindcasts/"
# hindcast_source = {'data_source_type': 'multiple_daily_nc_files',
#                    'data_source': hindcast_folder}
# hindcast_folder = "data/hindcast_test/"
# hindcast_source = {'data_source_type': 'single_nc_file',
#                    'data_source': hindcast_folder}
hindcast_source = {'data_source_type': 'cop_opendap',
                   'data_source': {'USERNAME': 'mmariuswiggert', 'PASSWORD': 'tamku3-qetroR-guwneq',
                                   'DATASET_ID': 'global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh'}}
# hindcast_source = {'data_source_type': 'cop_opendap',
#                    'data_source': {'USERNAME': 'mmariuswiggert', 'PASSWORD': 'tamku3-qetroR-guwneq',
#                                    'DATASET_ID': 'global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh'}}
# datetime.datetime(2021, 12, 1, 12, 10, 10, tzinfo=datetime.timezone.utc)

forecast_folder = "data/forecast_test/"
forecasts_source = {'data_source_type': 'single_nc_file',
                   'data_source': forecast_folder}

plan_on_gt=True
prob = Problem(x_0, x_T, t_0,
               platform_config_dict=platform_config_dict,
               hindcast_source= hindcast_source,
               forecast_source=forecasts_source,
               plan_on_gt = plan_on_gt,
               x_T_radius=0.1)
#%%
prob.viz(time=datetime.datetime(2021, 11, 26, 10, 10, 10, tzinfo=datetime.timezone.utc),
    cut_out_in_deg=1.2, plot_start_target=True) # plots the current at t_0 with the start and goal position
# # create a video of the underlying currents rendered in Jupyter, Safari or as file
# prob.viz(video=True) # renders in Jupyter
# prob.viz(video=True, html_render='safari')    # renders in Safari
# prob.viz(video=True, filename='problem_animation.gif', temp_horizon_viz_in_h=200) # saves as gif file
#%% Check feasibility of the problem
feasible, earliest_T, gt_run_simulator = utils.check_feasibility_2D_w_sim(
    problem=prob, T_hours_forward=95, deg_around_xt_xT_box=1.2, sim_dt=7200,
    conv_m_to_deg=111120, grid_res=[0.04, 0.04], hours_to_hj_solve_timescale=3600,
    sim_deg_around_x_t=4)
# 67.0 exactly!
#%%
plt.rcParams['font.size'] = '40'
plt.rcParams["font.family"] = "Palatino"
#%
# ax = prob.viz(cut_out_in_deg=1.2, return_ax=True, alpha_currents=0.2, plot_start_target=False)
ax = gt_run_simulator.high_level_planner.plot_reachability_snapshot(
    rel_time_in_h=5, multi_reachability=True, granularity_in_h=10., alpha_color=1.,
    time_to_reach=True, return_ax=True, fig_size_inches=(4*3.5, 4*2.5))
plt.show()

#%% Make the colorbar fit finally
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

im = plt.imshow(np.arange(200).reshape((20, 10)))
add_colorbar(im)

#%% plot 3D
from scipy.interpolate import interp1d
rel_time_in_h = 27
# get_initial_value
initial_values = gt_run_simulator.high_level_planner.get_initial_values(direction=gt_run_simulator.high_level_planner.specific_settings['direction'])

# interpolate
val_at_t = interp1d(gt_run_simulator.high_level_planner.reach_times - gt_run_simulator.high_level_planner.reach_times[0], gt_run_simulator.high_level_planner.all_values, axis=0, kind='linear')(
    rel_time_in_h * gt_run_simulator.high_level_planner.specific_settings['hours_to_hj_solve_timescale']).squeeze()

val_below_0 = np.where(val_at_t > 0, np.NaN, val_at_t)
# hj.viz.visFunc(gt_run_simulator.high_level_planner.grid,
#                val_at_t, color='black', alpha=0.5, marker=None)

grid = gt_run_simulator.high_level_planner.grid
data = val_below_0
color='black'
alpha=0.5
#%%
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go
fig = go.Figure(
        data=[go.Surface(opacity=alpha,
            x=grid.states[:, :, 0],
            y=grid.states[:, :, 1],
            z=data, connectgaps=False)])
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))


fig.show()

#%%
gt_run_simulator.high_level_planner.plot_reachability_animation(
    type='mp4', multi_reachability=True, granularity_in_h=5, time_to_reach=True)
#%% Plot the time-optimal trajectory (if true currents are known)
# gt_run_simulator.plot_trajectory(plotting_type='2D')
# gt_run_simulator.plot_trajectory(plotting_type='ctrl')
gt_run_simulator.plot_trajectory(plotting_type='2D_w_currents_w_controls')
#%%
gt_run_simulator.create_animation(vid_file_name="simulation_animation.gif", deg_around_x0_xT_box=1.2,
                     temporal_stride=6, time_interval_between_pics=200,
                     linewidth=4, marker=None, linestyle='-', add_ax_func=None)
#%% Set-Up and run Simulator
sim = OceanNavSimulator(sim_config_dict="simulator.yaml",
                        control_config_dict='reach_controller.yaml',
                        problem=prob)
start = time.time()
sim.run(T_in_h=10)
print("Took : ", time.time() - start)
print("Arrived after {} h".format((sim.trajectory[3,-1] - sim.trajectory[3, 0])/3600))
# 67.0 exactly!
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
gt_run_simulator.high_level_planner.plot_reachability_snapshot(rel_time_in_h=0,
                                                               multi_reachability=True,
                                                               granularity_in_h=5.,
                                                               time_to_reach=True)
#%%
import matplotlib.pyplot as plt
self = gt_run_simulator.high_level_planner
# gt_run_simulator.high_level_planner.vis_Value_func_along_traj()
figsize=(12,12)
return_ax=False
extra_traj=None
time_to_reach=True
fig, ax = plt.subplots(figsize=figsize)

if time_to_reach:
    all_values_dimensional = 1 + self.all_values - (self.reach_times/self.reach_times[-1]).reshape(-1, 1, 1)
    all_values = all_values_dimensional*self.specific_settings['T_goal_in_h']
    ylabel = "Earliest-time-to-reach"
else:
    ylabel = r"$\phi(x_t)$"
    all_values = self.all_values

reach_times = (self.reach_times - self.reach_times[0])/self.specific_settings['hours_to_hj_solve_timescale']
traj_times = (self.planned_trajs[-1]['traj'][3,:] - self.current_data_t_0 - self.reach_times[0])/self.specific_settings['hours_to_hj_solve_timescale']

hj.viz.visValFuncTraj(ax,
                      traj_times=traj_times,
                      x_traj=self.planned_trajs[-1]['traj'][:2,:],
                      all_times=reach_times,
                      all_values=all_values, grid=self.grid,
                      flip_times=False)
                      # ylabel=ylabel)
if extra_traj is not None:
    extra_traj_times = (extra_traj[3, :] - self.current_data_t_0 - self.reach_times[0]) / \
                 self.specific_settings['hours_to_hj_solve_timescale']
    hj.viz.visValFuncTraj(ax,
                          traj_times=extra_traj_times,
                          x_traj=extra_traj[:2, :],
                          all_times=self.reach_times/self.specific_settings['hours_to_hj_solve_timescale'],
                          all_values=all_values, grid=self.grid,
                          flip_times=False)
                          # ylabel=ylabel)
#%% Plot 2D reachable set evolution

#%%
sim.high_level_planner.plot_reachability_animation(type='mp4', multi_reachability=True, granularity_in_h=1, time_to_reach=True)