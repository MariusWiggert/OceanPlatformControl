# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import datetime
import json
import time
import os
from typing import Optional, Type, Dict

import pytz
import ray
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.evaluation import Episode
from ray.tune.logger import UnifiedLogger

from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.scripts.Utils import Utils


class RLRunner:
    def __init__(
        self,
        name: str,
        agent_class: Type,
        agent_config: dict,
        ocean_env_config: dict,
        feature_constructor_config: dict,
        model_config: dict,
        reward_function_config: dict,
        verbose: Optional[int] = 0,
    ):
        self.name = name
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.ocean_env_config = ocean_env_config
        self.feature_constructor_config = feature_constructor_config
        self.model_config = model_config
        self.reward_function_config = reward_function_config
        self.verbose = verbose
        self.results = []
        self.iterations = 0
        self.train_times = []

        # Step 1: Prepare Paths
        self.timestring = datetime.datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y_%m_%d_%H_%M_%S")
        self.results_folder = f'/seaweed-storage/experiments/{name}_{self.timestring}/'
        self.config_folder = f'{self.results_folder}config/'
        self.checkpoints_folder = f'{self.results_folder}checkpoints/'

        # Step 2: Save configuration
        Utils.ensure_storage_connection()
        os.makedirs(self.results_folder)
        os.makedirs(self.config_folder)
        Utils.clean_results(f'/seaweed-storage/experiments/', verbose=1)
        json.dump(self.agent_config,                open(f'{self.config_folder}agent_config.json', "w"), indent=4)
        json.dump(self.ocean_env_config,            open(f'{self.config_folder}ocean_env_config.json', "w"), indent=4)
        json.dump(self.feature_constructor_config,  open(f'{self.config_folder}feature_constructor_config.json', "w"), indent=4)
        json.dump(self.model_config,                open(f'{self.config_folder}model_config.json', "w"), indent=4)
        json.dump(self.reward_function_config,      open(f'{self.config_folder}reward_function_config.json', "w"), indent=4)

        # Step 3: Register Env
        # env_config: env_config.num_workers, env_config.worker_index, env_config.vector_index, env_config.remote
        self.agent_config["env"] = "OceanEnv"
        ray.tune.registry.register_env("OceanEnv", lambda env_config: OceanEnv(
            worker_index=env_config.worker_index,
            config= self.ocean_env_config | {
                'experiments_folder': self.results_folder,
                'feature_constructor_config': self.feature_constructor_config,
                'reward_function_config': self.reward_function_config,
            },
            verbose=self.verbose-1
        ))

        # Step 4: Register Model
        # https://docs.ray.io/en/latest/rllib/package_ref/models.html
        # ModelCatalog.register_custom_model("OceanNNModel", OceanNNModel)

        # Step 5: Success Metric
        # https://docs.ray.io/en/latest/rllib/rllib-training.html#callbacks-and-custom-metrics
        # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        class CustomCallback(ray.rllib.agents.callbacks.DefaultCallbacks):
            def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int):
                info = episode.last_info_for()
                if info['problem_status'] != 0:
                    episode.custom_metrics["success"] = info['problem_status'] > 0
                if info['problem_status'] > 0:
                    episode.custom_metrics["arrival_time_in_h"] = info["arrival_time_in_h"]
        self.agent_config["callbacks"] = CustomCallback

        self.agent = agent_class(self.agent_config, logger_creator=lambda config: UnifiedLogger(config, self.results_folder, loggers=None))

    def run(self, iterations = 100, silent=False):
        print(f"Starting training with {iterations} iterations:")

        for iteration in range(1, iterations + 1):
            train_start = time.time()
            result = self.agent.train()
            self.train_times.append(time.time() - train_start)

            Utils.ensure_storage_connection()
            os.makedirs(self.checkpoints_folder, exist_ok=True)
            self.agent.save(self.checkpoints_folder)

            self.results.append(result)

            if self.verbose and not silent:
                self.print_result(result, iteration, iterations)

    def print_result(self, result, iteration, iterations):
        print(f'--------- Iteration {iteration} (Total Samples: {result["info"]["num_env_steps_trained"]}) ---------')

        print(f'-- Episode Rewards [Min: {result["episode_reward_min"]:.1f}, Mean: {result["episode_reward_mean"]:.2f}, Max: {result["episode_reward_max"]:.1f}]  --')
        print(f'[{", ".join([f"{elem:.1f}" for elem in result["hist_stats"]["episode_reward"][-min(50, result["episodes_this_iter"]):]])}]')
        print(' ')

        episodes_this_iteration = result["hist_stats"]["episode_lengths"][-result["episodes_this_iter"]:]
        print(f'-- Episode Length [Min: {min(episodes_this_iteration):.0f}, Mean: {result["episode_len_mean"]:.1f}, Max: {max(episodes_this_iteration):.0f}] --')
        print(result["hist_stats"]["episode_lengths"][-min(50, result["episodes_this_iter"]):])
        print(f'Episodes: {len(episodes_this_iteration)}')
        print(f'Episode Steps:  {sum(episodes_this_iteration)}')
        print(f'Training Samples: {result["num_env_steps_trained_this_iter"]}')
        print(' ')

        print('-- Timing --')
        train_time = self.train_times[-1]
        print(f'Iteration Time: {train_time/60:.2f}min ({iterations * (train_time) / 60:.1f}min for {iterations} iterations, {(iterations - iteration) * (train_time) / 60:.1f}min to go)')
        # pprint(result["sampler_perf"])
        # print(f'total time per step: {sum(result["sampler_perf"].values()):.2f}ms')