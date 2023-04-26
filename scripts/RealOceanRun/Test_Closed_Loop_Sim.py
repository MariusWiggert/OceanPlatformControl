from matplotlib import pyplot as plt

from ocean_navigation_simulator.utils.misc import get_c3
import numpy as np
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.problem_factories.Constructor import Constructor
from ocean_navigation_simulator.environment.PlatformState import SpatioTemporalPoint
from ocean_navigation_simulator.environment.Platform import PlatformAction
import os
import pickle
import logging
import datetime
from ocean_navigation_simulator.utils.misc import set_arena_loggers
set_arena_loggers(logging.DEBUG)
c3 = get_c3()

# get the real ocean run object
this = c3.RealOceanSimRun.get('Ocean_Beach_to_HFB_NOAA_April_Test_api_test_mission_HJ_controller_onFMRC_T_horizon_24h')
this = this.get("mission.missionConfig, mission.experiment.timeout_in_sec," +
            "mission.experiment.arenaConfig, mission.experiment.objectiveConfig," +
            "controllerSetting.ctrlConfig, observerSetting.observerConfig")

#%% Step 1: Set_up code
# Set up file paths and download folders
temp_folder = '/tmp/' + this.id + '/'
# set download directories (ignore set ones in arenaConfig)
arenaConfig = this.mission.experiment.arenaConfig
arenaConfig['timeout'] = this.mission.experiment.timeout_in_sec
to_download_forecast_files = True

# for forecast
if arenaConfig['ocean_dict']['forecast'] is not None:
    arenaConfig['ocean_dict']['forecast']['source_settings']['folder'] = '/tmp/forecast_files/'
    to_download_forecast_files = arenaConfig["ocean_dict"]["forecast"]["source"] == "forecast_files"

# Making FC also true currents
arenaConfig['ocean_dict']['hindcast'] = {'field': 'OceanCurrents',
                                         'source': 'hindcast_files',
                                         'source_settings': {
                                             'folder': '/tmp/forecast_files/'}}
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
ctrl_conf['T_goal_in_seconds'] = 3600 * 70
#%% Get objects for closed-loop simulation
# Step 0: Create Constructor object which contains arena, problem, controller and observer
constructor = Constructor(
    arena_conf=arenaConfig,
    mission_conf=this.mission.missionConfig,
    objective_conf=this.mission.experiment.objectiveConfig,
    ctrl_conf=this.controllerSetting.ctrlConfig,
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

#%% Run first planning
action = controller.get_action(observation=observation)
# %% Various plotting of the reachability computations
ax = controller.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=True,
    return_ax=True,
)
problem.plot(ax=ax)
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
# controller.plot_reachability_snapshot_over_currents(rel_time_in_seconds=0, granularity_in_h=5, time_to_reach=False)
# controller.plot_reachability_animation(time_to_reach=False, granularity_in_h=5, filename="test_reach_animation.mp4")
# controller.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, with_opt_ctrl=True,
#                                     filename="test_reach_animation_w_ctrl.mp4", forward_time=True)
#%% Run closed-loop simulation
while problem_status == 0:
    # Get action
    action = controller.get_action(observation=observation)

    # execute action
    observation = arena.step(action)

    # update problem status
    problem_status = arena.problem_status(problem=problem)
# TODO: currently it strands so terminates because of that...
#%% Additional plotting and animation lines
arena.plot_all_on_map(problem=problem, margin=0.1)

#%% For debugging: Plot the currents over that area and time generally
t_start = datetime.datetime.fromisoformat(this.mission.missionConfig['x_0'][0]['date_time'])
def add_prob(ax, time):
    problem.plot(ax=ax)
arena.ocean_field.hindcast_data_source.animate_data(
        x_interval = [-123, -122],
        y_interval= [37, 38],
        t_interval = [t_start, t_start + datetime.timedelta(hours=60)],
        vmax=0.4, vmin=0.0,
        add_ax_func=add_prob
)