import sys
sys.path.extend(['/home/ubuntu/OceanPlatformControl'])
print('Python %s on %s' % (sys.version, sys.platform))
print(sys.path)

import os
os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'

import datetime
import time
import pytz
import ray
from ray.rllib.agents.dqn.apex import ApexTrainer

from ocean_navigation_simulator.reinforcement_learning_scripts.RLRunner import RLRunner
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils


print(f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}')
script_start_time = time.time()

Utils.ray_init()
Utils.ensure_storage_connection()

runner = RLRunner(
    name='baseline_hj_planner_hindcast',
    scenario_name='gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast',
    agent_class=ApexTrainer,
    agent_config={
        ## Framework
        "framework": "tf2",
        "eager_tracing": True,
        ## Episodes
        # "batch_mode": "complete_episodes",
        # "horizon": 10000,
        # "soft_horizon": True,
        # "no_done_at_end": False,
        ## Model
        "hiddens": [64, 64],
        ## DQN
        "num_atoms": 1,
        "n_step": 1,
        "dueling": True,
        "double_q": True,
        ## Training
        "replay_buffer_config": {
            "learning_starts": 50000,
        },
        "train_batch_size": 512,
        "rollout_fragment_length": 100,
        ## Workers
        "num_gpus": 1,
        "num_workers": 102,
        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 0,
        "placement_strategy": "SPREAD",
        "ignore_worker_failures": True,
        "recreate_failed_workers": True,
    },
    ocean_env_config={
        'generation_folder': '/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/fixed_forecast_50000_batches/',
        'scenario_name': 'gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast',
        'arena_steps_per_env_step': 1,
        'actions': 8,
        'render': False,
        'fake': 'hj_planner_hindcast', #one of: False, 'random', 'naive, 'hj_planner_forecast', 'hj_planner_hindcast'
    },
    feature_constructor_config={
        'num_measurements': 0,
        'map': {
            'xy_width_degree': 0.5,
            'xy_width_points': 5,
            'ttr_forecast': True,
            'ttr_hindcast': False,
            'observer': [], # list from: error_u, error_v, std_error_u, std_error_v, initial_forecast_u, initial_forecast_v, water_u, water_v
        },
    },
    model_config={
        'use_custom': True,
        'hidden_units': [64, 64],
    },
    reward_function_config={
        'target_bonus': 0,
        'fail_punishment': 0,
    },
    verbose=2
)
# 10 iterations ~ 42min (6min + 9 * 4min) @ 70 cores, 500MB Animations
# 100 iterations ~ 420min = 7h, 5GB Animation
runner.run(iterations=100)

ray.shutdown()

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")