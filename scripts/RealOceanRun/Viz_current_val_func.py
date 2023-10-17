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

# get the real ocean run object
this = c3.RealOceanSimRun.get('Ocean_Beach_to_HFB_NOAA_u_0.2mps_2023_4_27_OB_to_HMB_HJ_controller_T_horizon_60h')
this = this.get("mission.missionConfig, mission.experiment.timeout_in_sec," +
            "mission.experiment.arenaConfig, mission.experiment.objectiveConfig," +
            "controllerSetting.ctrlConfig, observerSetting.observerConfig," +
            "recentPlatformState, trajectory, valueFunctionInfo, valueFunction, oceanSimResult," +
            "status, planningStatus, actionStatus"
            )
#%% Step 1: Print Status
print("status: ", this.status)
print("planningStatus: ", this.planningStatus)
print("actionStatus: ", this.actionStatus)
print("recentPlatformState: ", this.recentPlatformState)
print("valueFunctionInfo: ", this.valueFunctionInfo)
print("oceanSimResult: ", this.oceanSimResult)

#%% Step 2: Download trajectory so far
# format [lon, lat, posix_time, u_magnitude, u_direction, error]
trajectory = this.trajectory.toStream().collectAndMerge().data
#TODO: viz over currents on map
#%% Step 3: load most recent value function
local_file = c3.HycomUtil.download_file_to_local(url=this.valueFunction.url, local_folder="tmp")
import xarray as xr
hj_val_func = xr.open_dataset(local_file)
#%% Step 3.1 Visualize value function!
time_idx = 10
print(datetime.datetime.fromtimestamp(hj_val_func['time'][time_idx]))
hj_val_func.isel(time=time_idx)['HJValueFunc'].T.plot(cmap='jet_r')
import matplotlib.pyplot as plt
plt.show()
# TODO: visualize properly
#%%