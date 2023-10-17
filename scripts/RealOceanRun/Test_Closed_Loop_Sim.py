import datetime
import logging
import os
import pickle

import numpy as np
from hj_reachability import ControlAndDisturbanceAffineDynamics
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.PlatformState import (
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.problem_factories.Constructor import (
    Constructor,
)
from ocean_navigation_simulator.utils.misc import get_c3, set_arena_loggers

set_arena_loggers(logging.DEBUG)
c3 = get_c3()
#%
# get the real ocean run object
this = c3.RealOceanSimRun.get('OceanExp_FC_NOAA_u_0_2mps_2023_5_03_HMB_to_OB_HJ_controller_T_horizon_60h')
this = this.get("mission.missionConfig, mission.experiment.timeout_in_sec," +
            "mission.experiment.arenaConfig, mission.experiment.objectiveConfig," +
            "controllerSetting.ctrlConfig, observerSetting.observerConfig")

#% Step 1: Set_up code
# Set up file paths and download folders
temp_folder = '/tmp/' + this.id + '/'
# set download directories (ignore set ones in arenaConfig)
arenaConfig = this.mission.experiment.arenaConfig
arenaConfig['timeout'] = this.mission.experiment.timeout_in_sec
to_download_forecast_files = True

# for forecast
if arenaConfig['ocean_dict']['forecast'] is not None:
    arenaConfig['ocean_dict']['forecast']['source_settings']['folder'] = 'data/noaa_forecast_files/'
    to_download_forecast_files = arenaConfig["ocean_dict"]["forecast"]["source"] == "forecast_files"
#%
# Making FC also true currents
# arenaConfig['ocean_dict']['hindcast'] = {'field': 'OceanCurrents',
#                                          'source': 'hindcast_files',
#                                          'source_settings': {
#                                              'folder': 'data/noaa_forecast_files/',
#                                               'source': 'hindcast_files'
#                                          }}
# NOTE: when we want to run it with what is set as ground truth, run this.
# arenaConfig['ocean_dict']['hindcast']['source_settings']['folder'] = '/tmp/hindcast_files/'

# prepping the file download
point_to_check = SpatioTemporalPoint.from_dict(this.mission.missionConfig['x_0'][0])
t_interval = [point_to_check.date_time,
              point_to_check.date_time + datetime.timedelta(
                  seconds=this.mission.experiment.timeout_in_sec + arenaConfig['casadi_cache_dict'][
                      'time_around_x_t'] + 7200)]

# modify ctrl_conf to extract optimal traj after planning
ctrl_conf=this.controllerSetting.ctrlConfig
ctrl_conf['calc_opt_traj_after_planning'] = True
miss_config = this.mission.missionConfig
print(miss_config)
#%%
# Start at HMB
# then go to 37.3, -122.5 {'lat': 37.35, 'lon': -122.5}
# then go back to HMB
# miss_config = {'target_radius': 0.02,
#                'x_0': [{'date_time': '2023-05-06T19:00:00+00:00', 'lat': 37.482812,'lon': -122.53145}],
#                'x_T': {'lat': 37.35, 'lon': -122.5}}
miss_config = {'target_radius': 0.02,
               'x_0': [{'date_time': '2023-05-07T19:00:00+00:00', 'lat': 37.35, 'lon': -122.5}],
               'x_T': {'lat': 37.482812,'lon': -122.53145}}
arenaConfig = {'casadi_cache_dict': {'deg_around_x_t': 0.3, 'time_around_x_t': 86400.0},
 'platform_dict': {'battery_cap_in_wh': 400.0,
  'u_max_in_mps': 0.2,
  'motor_efficiency': 1.0,
  'solar_panel_size': 1.0,
  'solar_efficiency': 0.2,
  'drag_factor': 675.0,
  'dt_in_s': 600.0},
 'use_geographic_coordinate_system': True,
 'spatial_boundary': None,
 'ocean_dict': {'region': 'Region 1',
  'hindcast': {'field': 'OceanCurrents',
   'source': 'opendap',
   'source_settings': {'service': 'copernicus',
    'currents': 'total',
    'USERNAME': 'mmariuswiggert',
    'PASSWORD': 'tamku3-qetroR-guwneq',
    'DATASET_ID': 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i'}},
  'forecast': None
   #              {'field': 'OceanCurrents',
   # 'source': 'forecast_files',
   # 'source_settings': {'source': 'noaa',
   #  'type': 'forecast',
   #  'folder': 'data/noaa_forecast_files/'}}
                },
 'solar_dict': {'hindcast': None, 'forecast': None},
 'seaweed_dict': {'hindcast': None, 'forecast': None},
 'timeout': 432000}

# ctrl_conf = {'T_goal_in_seconds': 216000,
#  'accuracy': 'high',
#  'artificial_dissipation_scheme': 'local_local',
#  'ctrl_name': 'ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner',
#  'deg_around_xt_xT_box': 0.3,
#  'direction': 'forward',
#  'grid_res': 0.005,
#  'n_time_vector': 100,
#  'obstacle_dict': {'obstacle_value': 1,
#   'path_to_obstacle_file': 'ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.004_max_elevation_0_northern_california.nc',
#   'safe_distance_to_obstacle': 0},
#  'progress_bar': True,
#  'replan_every_X_seconds': None,
#  'replan_on_new_fmrc': True,
#  'use_geographic_coordinate_system': True,
#             'initial_set_radii': [0.02, 0.02],
#  'calc_opt_traj_after_planning': False}
#%
# #%% hack to get it to download past forecasts
# miss_config['x_0'][0]['date_time'] = '2023-05-03T19:00:00+00:00'
#%
#% Get objects for closed-loop simulation
# Step 0: Create Constructor object which contains arena, problem, controller and observer
constructor = Constructor(
    arena_conf=arenaConfig,
    mission_conf=miss_config,
    objective_conf=this.mission.experiment.objectiveConfig,
    ctrl_conf=ctrl_conf,
    observer_conf=this.observerSetting.observerConfig,
    c3=c3,
    download_files=False,
    timeout_in_sec=arenaConfig["timeout"],
    throw_exceptions=False,
)

# Step 1.1 Retrieve problem
problem = constructor.problem

# Step 1.2: Retrieve arena
arena = constructor.arena
observation = arena.reset(platform_state=problem.start_state)
problem_status = arena.problem_status(problem=problem)

# Step 2: Retrieve Controller
controller = constructor.controller

# Step 3: Retrieve observer
observer = constructor.observer
#% Run first planning
action = controller.get_action(observation=observation)
#%% Various plotting of the reachability computations
ax = controller.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=True,
    return_ax=True,
)
plt.show()
#%%
# problem.plot(ax=ax)
ax.plot(controller.x_traj[0,:], controller.x_traj[1,:], color='k', label="State Trajectory", linewidth=2)
ctrl_stride = 5
u_vec = controller.contr_seq[0, ::ctrl_stride] * np.cos(controller.contr_seq[1, ::ctrl_stride])
v_vec = controller.contr_seq[0, ::ctrl_stride] * np.sin(controller.contr_seq[1, ::ctrl_stride])
ax.quiver(
    controller.x_traj[0, :-1:ctrl_stride],
    controller.x_traj[1, :-1:ctrl_stride],
    u_vec,
    v_vec,
    color='magenta',
    scale=15,
    angles="xy",
    label="Control Inputs",
)
plt.show()
#%%
problem.start_state.date_time = problem.start_state.date_time + datetime.timedelta(hours=3)
observation = arena.reset(platform_state=problem.start_state)
#%%
# #%%
# controller.animate_value_func_3D()
#%
# controller.plot_reachability_snapshot_over_currents(rel_time_in_seconds=0, granularity_in_h=5, time_to_reach=False)
# controller.plot_reachability_animation(time_to_reach=False, granularity_in_h=5, filename="test_reach_animation.mp4")
# controller.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, with_opt_ctrl=True,
#                                     filename="test_reach_animation_w_ctrl.mp4", forward_time=True)
#% Run closed-loop simulation
while problem_status == 0:
    # Get action
    action = controller.get_action(observation=observation)

    # execute action
    observation = arena.step(action)

    # update problem status
    problem_status = arena.problem_status(problem=problem)
#% Additional plotting and animation lines
arena.plot_all_on_map(problem=problem, margin=0.1, vmax=0.5, vmin=0,
                      control_stride=8, control_vec_scale=8,
                      spatial_resolution=1/30)
#%
arena.animate_trajectory(margin=0.1,
                         ctrl_scale=8,
                         temporal_resolution=3600,
                         spatial_resolution=1/30,
                         vmax=0.5, vmin=0)
#%% Note:
took_h = (arena.state_trajectory[-1, 2] - arena.state_trajectory[0, 2]) / 3600
print("took {} hours".format(took_h))
#%% For debugging: Plot the currents over that area and time generally
t_start = datetime.datetime.fromisoformat(miss_config['x_0'][0]['date_time'])
def add_prob(ax, time):
    problem.plot(ax=ax)
# arena.ocean_field.hindcast_data_source.animate_data(
#         output= "hindcast.mp4",
#         x_interval = [-123, -122],
#         y_interval= [37, 38],
#         spatial_resolution=1/30,
#         t_interval = [t_start, t_start + datetime.timedelta(hours=60)],
#         vmax=0.5, vmin=0.0,
#         add_ax_func=add_prob
# )
arena.ocean_field.forecast_data_source.animate_data(
        output= "forecast.mp4",
        x_interval = [-123, -122],
        y_interval= [37, 38],
        spatial_resolution=1/30,
        temporal_resolution=3600,
        t_interval = [t_start, t_start + datetime.timedelta(hours=60)],
        vmax=0.5, vmin=0.0,
        add_ax_func=add_prob
)