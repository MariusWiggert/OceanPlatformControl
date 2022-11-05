#%%
import datetime
import logging
import os

import pandas as pd

from ocean_navigation_simulator.reinforcement_learning.missions.MissionGenerator import (
    MissionGenerator,
)
from ocean_navigation_simulator.reinforcement_learning.runners.GenerationRunner import (
    GenerationRunner,
)
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import set_arena_loggers


# set_arena_loggers(logging.INFO)

# change to this if basic setup works
set_arena_loggers(logging.DEBUG)
logging.getLogger("MissionGenerator").setLevel(logging.DEBUG)

# or this if you want to ignore all data warnings
# logging.getLogger("MissionGenerator").setLevel(logging.FATAL)

# TODO: some small issues with file downloading (Test with c3 cloud to figure out why)
# TODO: Merge selectively into the experimentRunner branch to get it to work there. -> almost done
# TODO: write C3 batch job for it, should be very light, probably just feeding in a json into the job.

config = {
    "scenario_file": "scripts/create_missions_example/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
    ##### Target Sampling #####
    # HYCOM HC: lon [-98.0,-76.4000244140625], lat[18.1200008392334,31.92000007629394]
    # Copernicus FC: lon: [-98.0, -76.416664], lat: [18.083334, 30.0]
    # Combined: [-98.0, -76.416664], [18.1200008392334, 30.0]
    "x_range": [units.Distance(deg=-95.9), units.Distance(deg=-78.52)],
    "y_range": [units.Distance(deg=20.22), units.Distance(deg=27.9)],
    "t_range": [
        # Copernicus FC: 2022-04 until today, HYCOM Hindcast: 2021-09 until today
        datetime.datetime(year=2022, month=8, day=20, tzinfo=datetime.timezone.utc),
        datetime.datetime(year=2022, month=8, day=28, tzinfo=datetime.timezone.utc),
    ],
    "problem_timeout": datetime.timedelta(hours=20),
    "target_distance_from_land": 0.5,
    "problem_target_radius": 0.1,
    ##### HJ Planner #####
    "hj_specific_settings": {
        # 'grid_res' has to be smaller than target_radius to prevent hj_solver errors
        "grid_res": 0.04,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
        "direction": "multi-time-reach-back",
        "n_time_vector": 199,  # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
        "accuracy": "high",
        "artificial_dissipation_scheme": "local_local",
        "run_without_x_T": True,
        "progress_bar": False,
        "use_geographic_coordinate_system": True,
        "d_max": 0,
        # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
        "replan_every_X_seconds": False,
        "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
        "T_goal_in_seconds": 3600 * 24 * 4,
    },
    "hj_planner_box": 2.0,
    ##### Start Sampling #####
    "feasible_missions_per_target": 8,
    "random_missions_per_target": 8,
    "min_distance_from_hj_frame": 0.5,
    "min_distance_from_land": 0.5,
    "feasible_mission_time": [datetime.timedelta(hours=5), datetime.timedelta(hours=24)],
    "random_min_distance_from_target": 0.5,
    ##### Actions #####
    "plot_batch": False,
    "animate_batch": False,
    "cache_forecast": False,
    "cache_hindcast": False,
}

results_folder = "/tmp/missions/"
os.makedirs(results_folder, exist_ok=True)
all_problems = []
for worker in range(1):
    mission_generator = MissionGenerator(
        config=config
        | {
            "seed": 2022 + worker,
            "cache_folder": results_folder + str(worker) + "_",
        }
    )
    problems, _, _ = mission_generator.cache_batch()
    all_problems.extend(problems)

df = pd.DataFrame([problem.to_dict() for problem in all_problems])
df.to_csv(results_folder + "problems.csv")

#%%
df.to_csv("problems.csv")
# GenerationRunner.plot_starts_and_targets(
#     results_folder,
#     scenario_file="scripts/create_missions_example/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
# )
# GenerationRunner.plot_target_dates_histogram(results_folder)
# GenerationRunner.plot_ttr_histogram(results_folder)
