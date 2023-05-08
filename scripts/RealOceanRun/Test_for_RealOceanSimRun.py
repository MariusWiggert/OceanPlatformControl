import datetime
import time

from ocean_navigation_simulator.utils.misc import get_c3

c3 = get_c3()

#%% load it if already exists
experimentName = "Ocean_Beach_to_HFB_NOAA_April_Test"
exp = c3.Experiment.get(experimentName)
# exp.remove()
#%% Step 1: Set-up an Experiment, Controller, and Mission (before operation)
## Create Experiment
# Forecast System that we can use:
# - HYCOM Global (daily forecasts, xh resolution, 1/12 deg spatial resolution)
# - Copernicus Global (daily forecasts for 5 days out, xh resolution, 1/12 deg spatial resolution)
# - NOAA West Coast Nowcast System (daily nowcasts for 24h, xh resolution, 10km spatial resolution)
#  Step 1: Set up an Experiment (collection of missions and controllers, all with same platform specs (thrust))
# Note: It gives a deserialize error, but it creates the object and everything works as expected.
experimentName = "Ocean_Beach_to_HFB_NOAA_April_Test"
max_speed_of_platform_in_meter_per_second = 0.1
forecast_system_to_use = "noaa"  # either of ['HYCOM', 'Copernicus', 'noaa']

# Configs for Experiment
arena_config = {
    "casadi_cache_dict": {
        "deg_around_x_t": 0.5,
        "time_around_x_t": 86400.0,
    },  # This is 24h in seconds!
    "platform_dict": {
        "battery_cap_in_wh": 400.0,
        "u_max_in_mps": max_speed_of_platform_in_meter_per_second,
        "motor_efficiency": 1.0,
        "solar_panel_size": 1.0,
        "solar_efficiency": 0.2,
        "drag_factor": 675.0,
        "dt_in_s": 600.0,
    },
    "use_geographic_coordinate_system": True,
    "spatial_boundary": None,
    "ocean_dict": {
        "region": "Region 1",  # This is the region of northern California
        "hindcast": {
            "field": "OceanCurrents",
            "source": "opendap",
            "source_settings": {
                "service": "copernicus",
                "currents": "total",
                "USERNAME": "mmariuswiggert",
                "PASSWORD": "tamku3-qetroR-guwneq",
                "DATASET_ID": "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
            },
        },
        "forecast": {
            "field": "OceanCurrents",
            "source": "forecast_files",
            "source_settings": {"source": forecast_system_to_use, "type": "forecast"},
        },
    },
    "solar_dict": {"hindcast": None, "forecast": None},
    "seaweed_dict": {"hindcast": None, "forecast": None},
}
# For Navigation Missions (go from A to B) always this Config. Others for Maximizing Growth.
objectiveConfig = {"type": "nav"}

# add exp
exp = c3.Experiment.createNew(
    name=experimentName,
    description="description of experiment",
    arenaConfig=arena_config,
    objectiveConfig=objectiveConfig,
    timeout_in_sec=3600 * 24 * 2,
)
exp = exp.get("experimentName")
## Step 2: Create Observer and Controller
# %% Ad Observer (currently None)
obs_name = "NoObserver"
exp.addObserver(name=obs_name, observerConfig={"observer": None})
# %% add controller
# Config for HJ Controller
ctrl_name = "HJ_controller_onFMRC_T_horizon_24h"
ctrl_config = {
    "ctrl_name": "ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner",
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",
    "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
    "T_goal_in_seconds": 3600 * 24,
    "grid_res": 0.01,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
}

exp.addController(name=ctrl_name, ctrlConfig=ctrl_config)
#%% Add Mission (A -> B)
print("current UTC datetime is: ", datetime.datetime.now())
mission_name = "api_test_mission"
missionConfig = {
    "target_radius": 0.1,  # in degrees
    "x_0": [
        {"date_time": "2023-04-26T15:00:00+00:00", "lat": 37.738160, "lon": -122.545469}
    ],  # the start position and time
    "x_T": {"lat": 37.482812, "lon": -122.531450},
}  # the target position

exp.addMission(name="api_test_mission", missionConfig=missionConfig)
#%% Create RealOceanSimRun
mission_id = experimentName + "_" + mission_name
controller_id = experimentName + "_" + ctrl_name
observer_id = experimentName + "_" + obs_name
real_oc_run = exp.createRealOceanRun(
    mission_id=mission_id, controller_id=controller_id, observer_id=observer_id
)
#%% Check if it is rdy to work
# The platform calls this to get the optimal thurst vector at each point in time
# The server will log both the lat, lon, time of the input and the thrust it computed
# Takes ≈2-3s (first call can take up to 60s because it has to install the python runtime in a VM)
real_oc_run = c3.RealOceanSimRun.get(
    "Ocean_Beach_to_HFB_NOAA_April_Test_api_test_mission_HJ_controller_onFMRC_T_horizon_24h"
)
start = time.time()
thrust_vec = real_oc_run.getThrustVector(
    lat=37.738160,
    lon=-122.545469,
    posix_time=datetime.datetime.fromisoformat("2023-04-26T15:00:00+00:00").timestamp(),
)
print(thrust_vec)
print("took seconds: ", time.time() - start)
#%% Step 4 Change status of the RealOceanSimRun
real_oc_run.updateStatus(new_status="inactive")
