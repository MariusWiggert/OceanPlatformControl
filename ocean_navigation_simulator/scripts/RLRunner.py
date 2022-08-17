import datetime
import json
import pickle
import shutil
import time
import os
from typing import Optional
import pprint

import ray
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
            "framework": "torch",
            ## Episodes
            "batch_mode": "truncate_episodes",
            "horizon": 720,
            "soft_horizon": True,
            "no_done_at_end": False,
            ## Environment
            "env": "OceanEnv",
            ## Workers
            "num_gpus": 0,
            "num_workers": 1,
            "num_cpus_per_worker": 1,
            "num_gpus_per_worker": 0,
            "placement_strategy": "SPREAD",
            "recreate_failed_workers": True,
            ## Model
            "hiddens": [64, 64],
            ## DQN
            "n_step": 1,
            "dueling": True,
            "double_q": True,
            ## Training
            "replay_buffer_config": {
                "learning_starts": 512,  # 50000
            },
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

        # Step 1: Prepare Paths
        timestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.results_folder = f'/seaweed-storage/experiments/{name}_{timestring}/'
        self.ray_results_folder = f'/seaweed-storage/ray_results/{name}_{timestring}/'

        RayUtils.check_storage_connection()
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.ray_results_folder, exist_ok=True)
        RayUtils.clean_ray_results(f'{self.ray_results_folder}..', verbose=1)

        # Step 2: Save configuration
        pickle.dump(self.agent_config, open(f'{self.results_folder}config.p', "wb"))
        json.dump(self.agent_config, open(f'{self.results_folder}config.json', "w"), indent=4, )

        # Step 3: Create Agent
        # env_config: env_config.num_workers, env_config.worker_index, env_config.vector_index, env_config.remote
        ray.tune.registry.register_env("OceanEnv", lambda env_config: OceanEnv(
            worker_index=env_config.worker_index,
            config={
                'feature_constructor_config': self.feature_constructor_config,
                'experiments_folder': self.results_folder,
                'generation_folder': '/seaweed-storage/generation_runner/gulf_of_mexico_HYCOM_hindcast/2_improved_planner/',
                'scenario_name': 'gulf_of_mexico_HYCOM_hindcast',
                'arena_steps_per_env_step': 1,
                'actions': 8,
            },
            verbose=100
        ))

        # ModelCatalog.register_custom_model("OceanNNModel", OceanNNModel)
        self.agent = ApexTrainer(self.agent_config, logger_creator=lambda config: UnifiedLogger(config, self.ray_results_folder, loggers=None))

    def run(self, iterations = 100, silent=False):
        print(f"Starting training with {iterations} iterations:")

        for iteration in range(1, iterations + 1):
            start = time.time()
            result = self.agent.train()
            iteration_time = time.time() - start

            self.results.append(result)

            if self.verbose and not silent:
                self.print_result(iteration, iterations, result, iteration_time)

        # pickle.dump(self.results, open(f'{self.results_folder}results.p', "wb"))
        # json.dump(self.results, open(f'{self.results_folder}results.json', "w"))

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

        # print('-- Timing --')
        # pprint(result["sampler_perf"])
        # print(f'total time per step: {sum(result["sampler_perf"].values()):.2f}ms')
        # print(f'iteration time: {iteration_time:.2f}s ({iterations * (iteration_time) / 60:.1f}min for {iterations} iterations, {(iterations - iteration) * (iteration_time) / 60:.1f}min to go)')