import sys
import yaml

sys.path.extend(['/home/ubuntu/OceanPlatformControl'])
print('Python %s on %s' % (sys.version, sys.platform))
print(sys.path)

import os
os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'

import datetime
import time
import pytz
from ray.rllib.agents.dqn.apex import ApexTrainer

from ocean_navigation_simulator.reinforcement_learning_scripts.RLRunner import RLRunner

print(f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}')
script_start_time = time.time()

with open(f'config/reinforcement_learning/training/experiment_basic.yaml') as f:
    experiment_config = yaml.load(f, Loader=yaml.FullLoader)

runner = RLRunner(
    # name='integrated_conv_3_3_3_fc_64_dueling_64_64_forecast_and_gp_42x42_1deg',
    name='test_config_file',
    scenario_name='gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast',
    agent_class=ApexTrainer,
    agent_config={
        # Environment
        "env": 'OceanEnv',
        ##### Framework #####
        "framework": "tf2",
        "eager_tracing": True,
        ##### Episodes #####
        # "batch_mode": "complete_episodes",
        # "horizon": 10000,
        # "soft_horizon": True,
        # "no_done_at_end": False,
        ##### Model #####
        "model": {
            "_use_default_native_models": True,
            # VisionNetwork (tf and torch): rllib.models.tf|torch.visionnet.py
            # These are used if no custom model is specified and the input space is 2D.
            # Filter config: List of [out_channels, kernel, stride] for each filter.
            # Example:
            # Use None for making RLlib try to find a default filter setup given the
            # observation space.
            # "conv_filters": [
            #     [16, 3, 1],
            #     [32, 3, 1],
            #     [32, 3, 1],
            # ],
            # Activation function descriptor.
            # Supported values are: "tanh", "relu", "swish" (or "silu"),
            # "linear" (or None).
            # "conv_activation": "relu",
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'tanh',
        },
        # dueling:
        "hiddens": [64, 64],
        ##### DQN #####
        "num_atoms": 1,
        "n_step": 1,
        "noisy": False,
        "dueling": True,
        "double_q": True,
        ##### ApeX #####
        "replay_buffer_config": {
            "learning_starts": 50000,
            # # For now we don't use the new ReplayBuffer API here
            # "_enable_replay_buffer_api": False,
            # "no_local_replay_buffer": True,
            # "type": "MultiAgentReplayBuffer",
            # "capacity": 2000000,
            # "replay_batch_size": 32,
            # "prioritized_replay_alpha": 0.6,
            # # Beta parameter for sampling from prioritized replay buffer.
            # "prioritized_replay_beta": 0.4,
            # # Epsilon to add to the TD errors when updating priorities.
            # "prioritized_replay_eps": 1e-6,
            # # prioritized_replay_alpha: Alpha parameter controls the degree of
            # #     prioritization in the buffer. In other words, when a buffer sample has
            # #     a higher temporal-difference error, with how much more probability
            # #     should it drawn to use to update the parametrized Q-network. 0.0
            # #     corresponds to uniform probability. Setting much above 1.0 may quickly
            # #     result as the sampling distribution could become heavily “pointy” with
            # #     low entropy.
            # #     prioritized_replay_beta: Beta parameter controls the degree of
            # #     importance sampling which suppresses the influence of gradient updates
            # #     from samples that have higher probability of being sampled via alpha
            # #     parameter and the temporal-difference error.
            # #     prioritized_replay_eps: Epsilon parameter sets the baseline probability
            # #     for sampling so that when the temporal-difference error of a sample is
            # #     zero, there is still a chance of drawing the sample.
        },
        "train_batch_size": 512,
        "rollout_fragment_length": 100,
        # "target_network_update_freq": 100000,
        # "timesteps_per_iteration": 25000,
        # "exploration_config": {"type": "PerWorkerEpsilonGreedy"},
        # "worker_side_prioritization": True,
        # # This will set the ratio of replayed from a buffer and learned
        # # on timesteps to sampled from an environment and stored in the replay
        # # buffer timesteps. Must be greater than 0.
        # "training_intensity": 1,
        ##### Workers #####
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
        'fake': False, #one of: False, 'random', 'naive, 'hj_planner_forecast', 'hj_planner_hindcast'
    },
    feature_constructor_config={
        'num_measurements': 0,
        'map': {
            'xy_width_degree': 0.2,
            'xy_width_points': 5,
            'flatten': False,
            'features': {
                'ttr_forecast': True,
                'ttr_hindcast': False,
                'observer': [], #['water_u', 'water_v'], # list from: 'error_u', 'error_v', 'std_error_u', 'std_error_v', 'initial_forecast_u', 'initial_forecast_v', 'water_u', 'water_v'
            }
        },
    },
    model_config={
        'use_custom': False,
        'q_hiddens': [64, 64],
        'add_layer_norm': False,
    },
    reward_function_config={
        'ttr_'
        'target_bonus': 0,
        'fail_punishment': 0,
    },
    verbose=2
)
# 10 iterations ~ 42min (6min + 9 * 4min) @ 70 cores, 500MB Animations
# 100 iterations ~ 420min = 7h, 5GB Animation
runner.run(epochs=100)

# ray.shutdown()


script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")