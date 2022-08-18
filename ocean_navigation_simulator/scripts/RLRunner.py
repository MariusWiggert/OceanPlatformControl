import datetime
import json
import pickle
import shutil
import time
import os
from typing import Optional, Type

import pytz
import ray
from ray.train import Trainer
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents.dqn.apex import ApexTrainer

from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.scripts.RayUtils import RayUtils


class RLRunner:
    def __init__(
        self,
        name: str,
        agent_class: Type,
        agent_config: dict,
        ocean_env_config: dict,
        feature_constructor_config: dict,
        reward_function_config: dict,
        verbose: Optional[int] = 0,
    ):
        self.name = name
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.ocean_env_config = ocean_env_config
        self.feature_constructor_config = feature_constructor_config
        self.reward_function_config = reward_function_config
        self.verbose = verbose
        self.results = []
        self.iterations = 0
        self.train_times = []

        # Step 1: Prepare Paths
        self.timestring = datetime.datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y_%m_%d_%H_%M_%S")
        self.results_folder = f'/seaweed-storage/experiments/{name}_{self.timestring}/'
        self.ray_results_folder = f'/seaweed-storage/ray_results/{name}_{self.timestring}/'
        self.checkpoints_folder = f'{self.results_folder}checkpoints/'

        # Step 2: Save configuration
        RayUtils.check_storage_connection()
        os.makedirs(self.results_folder)
        json.dump(self.agent_config,                open(f'{self.results_folder}agent_config.json', "w"), indent=4)
        json.dump(self.ocean_env_config,            open(f'{self.results_folder}ocean_env_config.json', "w"), indent=4)
        json.dump(self.feature_constructor_config,  open(f'{self.results_folder}feature_constructor_config.json', "w"), indent=4)
        json.dump(self.reward_function_config,      open(f'{self.results_folder}reward_function_config.json', "w"), indent=4)

        # Step 3: Register Env, Model and create Agent
        RayUtils.check_storage_connection()
        os.makedirs(self.ray_results_folder, exist_ok=True)
        RayUtils.clean_ray_results('/seaweed-storage/ray_results/', verbose=1)

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

        # ModelCatalog.register_custom_model("OceanNNModel", OceanNNModel)
        self.agent = ApexTrainer(self.agent_config, logger_creator=lambda config: UnifiedLogger(config, self.ray_results_folder, loggers=None))

    def run(self, iterations = 100, silent=False):
        print(f"Starting training with {iterations} iterations:")

        for iteration in range(1, iterations + 1):
            train_start = time.time()
            result = self.agent.train()
            self.train_times.append(time.time() - train_start)

            RayUtils.check_storage_connection()
            os.makedirs(self.checkpoints_folder, exist_ok=True)
            self.agent.save(self.checkpoints_folder)

            self.results.append(result)

            if self.verbose and not silent:
                self.print_result(result, iteration, iterations)

        RayUtils.check_storage_connection()
        # pickle.dump(self.results, open(f'{self.results_folder}results.p', "wb"))
        # json.dump(self.results, open(f'{self.results_folder}results.json', "w"))

    def print_result(self, result, iteration, iterations):
        print(f'--------- Iteration {iteration} (Total Samples: {result["info"]["num_env_steps_trained"]}) ---------')

        print(f'-- Episode Rewards [Min: {result["episode_reward_min"]:.2f}, Mean: {result["episode_reward_mean"]:.2f}, Max: {result["episode_reward_max"]:.2f}]  --')
        print(f'[{", ".join([f"{elem:.1f}" for elem in result["hist_stats"]["episode_reward"][-min(50, result["episodes_this_iter"]):]])}]')
        print(' ')

        episodes_this_iteration = result["hist_stats"]["episode_lengths"][-result["episodes_this_iter"]:]
        print(f'-- Episode Length [Min: {min(episodes_this_iteration):.2f}, Mean: {result["episode_len_mean"]:.2f}, Max: {max(episodes_this_iteration):.2f}] --')
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