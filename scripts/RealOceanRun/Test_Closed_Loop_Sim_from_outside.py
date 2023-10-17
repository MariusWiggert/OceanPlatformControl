import datetime
import logging

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

# get the real ocean run object
this = c3.RealOceanSimRun.get('Ocean_Beach_to_HFB_NOAA_u_0.2mps_2023_4_27_OB_to_HMB_HJ_controller_T_horizon_60h')
this = this.get(
    "mission.missionConfig, mission.experiment.timeout_in_sec,"
    + "mission.experiment.arenaConfig, mission.experiment.objectiveConfig,"
    + "controllerSetting.ctrlConfig, observerSetting.observerConfig"
)
## list all files in current directory
print("The files and folders in {} are:".format(os.getcwd()))
items = os.listdir(os.getcwd())
for item in items:
    print(item)
#%%
this.checkForReplanning()
#%% Step 1: Set_up code
# Set up file paths and download folders
temp_folder = "/tmp/" + this.id + "/"
# set download directories (ignore set ones in arenaConfig)
arenaConfig = this.mission.experiment.arenaConfig
arenaConfig["timeout"] = this.mission.experiment.timeout_in_sec
to_download_forecast_files = False

# for hindcast
arenaConfig["ocean_dict"]["hindcast"]["source_settings"]["folder"] = "/tmp/hindcast_files/"
# for forecast
if arenaConfig["ocean_dict"]["forecast"] is not None:
    arenaConfig["ocean_dict"]["forecast"]["source_settings"]["folder"] = "/tmp/forecast_files/"
    to_download_forecast_files = arenaConfig["ocean_dict"]["forecast"]["source"] == "forecast_files"

# prepping the file download
point_to_check = SpatioTemporalPoint.from_dict(this.mission.missionConfig["x_0"][0])
t_interval = [
    point_to_check.date_time,
    point_to_check.date_time
    + datetime.timedelta(
        seconds=this.mission.experiment.timeout_in_sec
        + arenaConfig["casadi_cache_dict"]["time_around_x_t"]
        + 7200
    ),
]

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
)

# Step 1.1 Retrieve problem
problem = constructor.problem

# Step 1.2: Retrieve arena
arena = constructor.arena
observation = arena.reset(platform_state=problem.start_state)
problem_status = arena.problem_status(problem=problem)
#%%
import numpy as np

platform_state = np.array(observation.platform_state.to_spatio_temporal_point()).tolist()
# # Trigger first replanning
this.checkForReplanning(current_datetime=problem.start_state.date_time)
last_replan_time = observation.platform_state.to_spatio_temporal_point().date_time

#%% Run closed-loop simulation
while problem_status == 0:
    # get platform state
    sim_time = observation.platform_state.date_time
    # trigger replanning every day (also multiple times to see that it does not replan all the time...)
    if sim_time - last_replan_time > datetime.timedelta(hours=12):
        print("check for replanning: ", sim_time)
        replaned = this.checkForReplanning(current_datetime=sim_time)
        print("replanned: ", replaned)
        print(this.get("valueFunctionInfo").valueFunctionInfo.FC_file_start)
        last_replan_time = sim_time

    # Get action via server interface...
    action_dict = this.getThrustVector(
        lat=observation.platform_state.lat.deg,
        lon=observation.platform_state.lon.deg,
        posix_time=observation.platform_state.date_time.timestamp(),
    )
    action = PlatformAction(**action_dict)

    # execute action
    observation = arena.step(action)

    # update problem status
    problem_status = arena.problem_status(problem=problem)

#%% Plotting
ax = arena.plot_all_on_map(problem=problem, return_ax=True)
#%% Now plot
