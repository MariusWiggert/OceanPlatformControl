import sys
sys.path.extend(['/home/ubuntu/OceanPlatformControl'])
print('Python %s on %s' % (sys.version, sys.platform))
print(sys.path)

import os
os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'

import datetime
import time
import pytz

from ocean_navigation_simulator.reinforcement_learning.scripts.GenerationRunner import GenerationRunner
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils
from ocean_navigation_simulator.utils import units

print(f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}')
script_start_time = time.time()

Utils.ray_init()

runner = GenerationRunner(
    name='verification_10000_problems',
    config={
        'scenario_name': 'gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast',
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
        'size': {
            'groups': 100,
            'batches_per_group': 100,
            'batch_size': 1,
        },
        'ray_options': {
            'max_retries': 10,
            'resources': {
                "CPU": 1.0,
                "GPU": 0.0,
                # RAM Min:  2238.0, RAM Mean: 3344.1, RAM Max:  3771.0
                "RAM": 4000,
                "Worker CPU": 1.0,
            }
        },
        'problem_factory_config': {
            'x_range': [units.Distance(deg=-92.5), units.Distance(deg=-89.5)],
            'y_range': [units.Distance(deg=23.5), units.Distance(deg=26.5)],
            't_range': [
                datetime.datetime(year=2022, month=6, day=1, tzinfo=datetime.timezone.utc),
                datetime.datetime(year=2022, month=6, day=11, tzinfo=datetime.timezone.utc)
            ],
            'mission_time_range': [datetime.timedelta(hours=24), datetime.timedelta(hours=36)],
            'problem_timeout': datetime.timedelta(hours=48),
            # 'problem_target_radius', 'goal_min_distance' have to be tuned with 'mission_time_range' for meaningful missions:
            #   - if 'goal_min_distance' is small in relation to 'target_radius' then the missions are extremely short
            #   - if
            #   0.1 deg ~= 11km = 3h @ 1m/s
            #   0.1 deg ~= 11km = 30h @ 0.1m/s
            #   1 deg ~= 111km = 30h @ 1m/s
            # if 'target_radius' is big the target is reached sooner than expected
            'start_distance_from_frame': 0.5,
            'target_distance_from_frame': 1.0,
            'problem_target_radius': 0.02,
            'target_min_distance': 0.15,
            # 'grid_res' has to be smaller than target_radius to prevent hj_solver errors
            'hj_planner': {
                'grid_res': 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
                # 'deg_around_xt_xT_box': 2.0,  # area over which to run HJ_reachability
            },
            'plot_batch': True,
        }
    },
    verbose=1,
)

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")