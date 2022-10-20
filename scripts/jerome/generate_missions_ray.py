import sys

sys.path.extend(["/home/ubuntu/OceanPlatformControl"])
print("Python %s on %s" % (sys.version, sys.platform))
print(sys.path)

import os

os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"

import datetime
import time

import pytz

from ocean_navigation_simulator.reinforcement_learning.runners.GenerationRunner import (
    GenerationRunner,
)
from ocean_navigation_simulator.utils import cluster_utils, units

print(
    f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}'
)
script_start_time = time.time()

cluster_utils.init_ray()

runner = GenerationRunner(
    name="divers_training",
    config={
        "scenario_file": "config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
        "generation_folder": "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast",
        # Only Hindcast:
        # -- GPU: 1 batch: 80s  = ~1.5min
        # Calculated: 1200 batches @ 8 gpus ~ 3.75h
        # -- CPU: 1 batch: 150s = ~3min
        # Measured: 96 batches @ 96 cores: 7min 17s
        # Measured: 384 batches @ 96 cores = 20min, 21min 37s
        # Measured: 10'000 batches @ 96 core = 9h 31min 33s
        # Calculated: 1'000 batches @ 96 core ~ 54min, 10'000 batches @ 96 core ~ 9h,
        # With Forecast:
        # Measured:
        # 1 batch @ 1 machine: 9min 47s -> 100 batches / 10min -> 600 batches / 1h -> 10'000 batches / 16.6h
        # 96 Batches = 8.5GB -> 5'000 batches = 442GB
        "size": {
            "groups": 1,
            "batches_per_group": 100,
        },
        "ray_options": {
            "max_retries": 10,
            "resources": {
                "CPU": 1.0,
                "GPU": 0.0,
                # RAM Min:  2238.0, RAM Mean: 3344.1, RAM Max:  3771.0
                "RAM": 4000,
                "Worker CPU": 1.0,
            },
        },
        "mission_generation": {
            ##### Target Sampling #####
            # Available: lon [-98.0,-76.4000244140625], lat[18.1200008392334,31.92000007629394]
            "x_range": [units.Distance(deg=-95.9), units.Distance(deg=-79.1)],
            "y_range": [units.Distance(deg=20.1), units.Distance(deg=29.9)],
            "t_range": [
                # Copernicus FC: 2022-04 until today
                # HYCOM Hindcast: 2021-09 until today
                datetime.datetime(year=2022, month=4, day=4, tzinfo=datetime.timezone.utc),
                datetime.datetime(year=2022, month=10, day=16, tzinfo=datetime.timezone.utc),
            ],
            "problem_timeout": datetime.timedelta(hours=150),
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
            },
            "hj_planner_box": 2.0,
            ##### Start Sampling #####
            "feasible_missions_per_target": 8,
            "random_missions_per_target": 8,
            "min_distance_from_hj_frame": 0.5,
            "min_distance_from_land": 0.5,
            "feasible_mission_time": [datetime.timedelta(hours=20), datetime.timedelta(hours=120)],
            "random_min_distance_from_target": 0.5,
            # 'target_radius', 'goal_min_distance' have to be tuned with 'mission_time_range' for meaningful missions:
            #   - if 'target_radius' is big the target is reached sooner than expected
            #   0.1 deg ~= 11km = 3h @ 1m/s, 30h @ 0.1m/s
            #   1 deg ~= 111km = 30h @ 1m/s, 300h @ 0.1m/s
            #   2 deg ~= 111km = 60h @ 1m/s

            "plot_batch": False,
            "animate_batch": True,
            "cache_forecast": False,
            "cache_hindcast": False,
        },
    },
    verbose=3,
)

script_time = time.time() - script_start_time
print(
    f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s."
)
