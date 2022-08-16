import datetime
import json
import pickle
import shutil
import time
from pprint import pprint
import os
from typing import Optional

import ray
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents.dqn.apex import ApexTrainer

from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.scripts.RayUtils import RayUtils


class RLRunner:
    def __init__(
        self,
        name: str,
        agent_class,
        agent_config: dict,
        feature_constructor_config = {},
        verbose: Optional[int] = 0
    ):
        self.name = name
        self.agent_class = agent_class
        self.agent_config = {
            ## Framework
            "framework": "tf2",
            ## Episodes
            "batch_mode": "truncate_episodes",
            "horizon": 720,
            "soft_horizon": True,
            "no_done_at_end": False,
            ## Environment
            "env": "OceanEnv",
            ## Workers
            "num_gpus": 1,
            "num_workers": 1,
            "num_cpus_per_worker": 1,
            "num_gpus_per_worker": 0,
            "placement_strategy": "SPREAD",
            ## Model
            "hiddens": [128, 128],
            ## DQN
            "n_step": 1,
            "dueling": True,
            "double_q": True,
            ## Training
            "learning_starts": 50000,
            "train_batch_size": 512,
            "rollout_fragment_length": 100,
        } | agent_config
        self.feature_constructor_config = {
            'num_measurements': 5,
            'ttr': {
                'xy_width_degree': 1,
                'xy_width_points': 10,
            },
        } | feature_constructor_config
        self.verbose = verbose
        self.results = []
        self.iterations = 0

        # Step 1: Prepare Folders
        self.results_folder = f'/seaweed-storage/experiments/{name}/'
        self.checkpoints_folder = self.results_folder + 'checkpoints/'
        self.plots_folder = self.results_folder + 'plots/'
        self.ray_results_base = f'/seaweed-storage/ray_results/'
        self.ray_results_folder = f'{self.ray_results_base}{name}/'


        # Step 1: Create and Clean Folders
        if os.path.exists(self.results_folder):
            shutil.rmtree(self.results_folder, ignore_errors=True)
        os.makedirs(self.results_folder , exist_ok=True)
        os.makedirs(self.checkpoints_folder , exist_ok=True)
        os.makedirs(self.plots_folder, exist_ok=True)
        if os.path.exists(self.ray_results_folder):
            shutil.rmtree(self.ray_results_folder, ignore_errors=True)
        os.makedirs(self.ray_results_folder, exist_ok=True)
        RayUtils.clean_ray_results(self.ray_results_base, verbose=1)

        # Step 2: Save configuration
        pickle.dump(self.agent_config, open(f'{self.results_folder}config.p', "wb"))
        json.dump(self.agent_config, open(f'{self.results_folder}config.json', "w"))

        # Step 3: Create Agent
        # env_config: Dict[num_workers, worker_index, vector_index]
        # and remote).
        def env_creator(env_config):
            print('Creating Env')
            return  OceanEnv(
                config={
                    'seed': env_config['worker_index'],
                    'feature_constructor_config': feature_constructor_config,
                },
                verbose=10
            )

        ray.tune.registry.register_env("OceanEnv", env_creator)
        RayUtils.init_ray()

        # ModelCatalog.register_custom_model("OceanNNModel", OceanNNModel)
        self.agent = ApexTrainer(agent_config, logger_creator=lambda config: UnifiedLogger(config, self.ray_results_folder, loggers=None))

    def run(self, iterations = 100, silent=False):
        print(f"Starting training with {iterations} iterations:")

        for iteration in range(1, iterations + 1):
            start = time.time()
            result = self.agent.train()
            iteration_time = time.time() - start

            self.results.append(result)

            if self.verbose and not silent:
                self.print_result(iteration, iterations, result, iteration_time)

        pickle.dump(self.results, open(f'{self.results_folder}results.p', "wb"))
        json.dump(self.results, open(f'{self.results_folder}results.json', "w"))

    def print_result(self, iteration, iterations, result, iteration_time):
        print(' ')
        print(' ')
        print(f'--------- Iteration {iteration} (total samples {result["info"]["num_env_steps_trained"]}) ---------')

        print('-- Episode Rewards --')
        print(f'[{", ".join([f"{elem:.1f}" for elem in result["hist_stats"]["episode_reward"][-min(25, result["episodes_this_iter"]):]])}]')
        print(f'Mean: {result["episode_reward_mean"]:.2f}')
        print(f'Max:  {result["episode_reward_max"]:.2f},')
        print(f'Min:  {result["episode_reward_min"]:.2f}')
        print(' ')

        print('-- Episode Length --')
        episodes_this_iteration = result["hist_stats"]["episode_lengths"][-result["episodes_this_iter"]:]
        print(result["hist_stats"]["episode_lengths"][-min(40, result["episodes_this_iter"]):])
        print(f'Mean: {result["episode_len_mean"]:.2f}')
        print(f'Min:  {min(episodes_this_iteration):.2f}')
        print(f'Max:  {max(episodes_this_iteration):.2f}')
        print(f'Number of Episodes: {len(episodes_this_iteration)}')
        print(f'Sum Episode Steps:  {sum(episodes_this_iteration)}')
        print(f'Samples for Training: {result["num_env_steps_trained_this_iter"]}')
        print(' ')

        print('-- Timing --')
        pprint(result["sampler_perf"])
        print(f'total time per step: {sum(result["sampler_perf"].values()):.2f}ms')
        print(f'iteration time: {iteration_time:.2f}s ({iterations * (iteration_time) / 60:.1f}min for {iterations} iterations, {(iterations - iteration) * (iteration_time) / 60:.1f}min to go)')