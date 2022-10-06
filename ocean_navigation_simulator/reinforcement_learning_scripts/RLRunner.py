import datetime
import json
import random
import time
import os
from typing import Optional, Dict, List
import numpy as np
import pytz
import ray
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents.dqn.apex import ApexTrainer
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
import wandb
import pprint
import tensorflow as tf
import torch

from ocean_navigation_simulator.reinforcement_learning import OceanKerasModel
from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.reinforcement_learning.OceanNNModel import OceanNNModel
from ocean_navigation_simulator.reinforcement_learning.OceanKerasModel import OceanKerasModel
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils
from ocean_navigation_simulator.utils.bcolors import bcolors

"""
    RLRunner takes a configuration to run a RL training with Rllib and hides away all the ugly stuff
    needed to run Rllib. It creates folder for results, saves configurations, registers Environment and
    Model with Rllib and finally prints results after each training iteration.
"""
class RLRunner:
    def __init__(
        self,
        name: str,
        config: dict,
        tags: Optional[List] = [],
        verbose: Optional[int] = 0,
    ):
        self.name = name
        self.config = config
        self.tags = tags
        self.verbose = verbose

        self.results = []
        self.iterations = 0
        self.train_times = []

        Utils.ray_init()

        # Step 1: Prepare Paths
        self.timestring = datetime.datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y_%m_%d_%H_%M_%S")
        self.experiment_folder = f'{self.config["experiments_folder"]}{self.name}_{self.timestring}/'
        self.config_folder = f'{self.experiment_folder}config/'
        self.checkpoints_folder = f'{self.experiment_folder}checkpoints/'

        # Step 2: Start Weights & Biases
        Utils.ensure_storage_connection()
        wandb.tensorboard.patch(
            root_logdir=self.experiment_folder,
            tensorboard_x=True
        )
        wandb.init(
            project="RL for underactuated navigation",
            entity="ocean-platform-control",
            dir="/seaweed-storage/",
            name=f'{self.name}',
            tags=["baseline" if self.config['environment']['fake'] else 'experiment'] + tags,
            config=self.config,
        )

        # Step 3: Save configuration & source
        Utils.ensure_storage_connection()
        # Utils.clean_results(self.config["experiments_folder"], verbose=1, delete=True)
        os.makedirs(self.experiment_folder)
        os.makedirs(self.config_folder)
        json.dump(self.config, open(f'{self.config_folder}config.json', "w"), indent=4)
        wandb.save(f'{self.config_folder}config.json')

        # Step 4: Save Source Files
        # with zipfile.ZipFile(f'{self.experiment_folder}source.zip', 'w') as file:
        #     file.write('config', 'config')
        #     file.write('ocean_navigation_simulation', 'ocean_navigation_simulation')
        #     file.write('scripts', 'scripts')
        #     file.write('setup', 'setup')

        # Step 5: Register Env
        # env_config: env_config.num_workers, env_config.worker_index, env_config.vector_index, env_config.remote
        ray.tune.registry.register_env("OceanEnv", lambda env_config: OceanEnv(
            config= self.config['environment'] | { 'experiment_folder': self.experiment_folder },
            feature_constructor_config=self.config['feature_constructor'],
            reward_function_config=self.config['reward_function'],
            worker_index=env_config.worker_index,
            verbose=self.verbose-1
        ))

        # Step 6: Register Model
        # Documentation:
        #   https://docs.ray.io/en/latest/rllib/package_ref/models.html
        #   https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
        # Usage:
        #   https://github.com/ray-project/ray/blob/releases/1.13.0/rllib/agents/dqn/distributional_q_tf_model.py
        #       https://github.com/ray-project/ray/blob/releases/1.13.0/rllib/models/tf/tf_modelv2.py
        #       https://github.com/ray-project/ray/blob/releases/1.13.0/rllib/models/modelv2.py
        #   https://github.com/ray-project/ray/blob/releases/1.13.0/rllib/agents/dqn/dqn_tf_policy.py
        if self.config['algorithm']['model'].get('custom_model', '') == 'OceanNNModel':
            ModelCatalog.register_custom_model("OceanNNModel", OceanNNModel)
        elif self.config['algorithm']['model'].get('custom_model', '') == 'OceanKerasModel':
            ModelCatalog.register_custom_model("OceanKerasModel", OceanKerasModel)

        # Step 7: Custom Metric
        # https://docs.ray.io/en/latest/rllib/rllib-training.html#callbacks-and-custom-metrics
        # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        class CustomCallback(ray.rllib.agents.callbacks.DefaultCallbacks):
            def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int):
                info = episode.last_info_for()
                if info['problem_status'] != 0:
                    episode.custom_metrics["success"] = info['problem_status'] > 0
                if info['problem_status'] > 0:
                    episode.custom_metrics["arrival_time_in_h"] = info["arrival_time_in_h"]
                episode.custom_metrics["problem_status"] = info["problem_status"]
                episode.custom_metrics["ram_usage_MB"] = info["ram_usage_MB"]
                episode.custom_metrics["episode_time"] = info["episode_time"]
                episode.custom_metrics["average_step_time"] = info["average_step_time"]
        self.config['algorithm']["callbacks"] = CustomCallback
        if self.config['environment']['fake']:
            self.config['algorithm']['explore'] = False

        # Step 8: Create Agent
        if self.config['algorithm_name'] == 'apex-dqn':
            agent_class = ApexTrainer

        # np.random.seed(self.config['algorithm']['seed'])
        # random.seed(self.config['algorithm']['seed'])
        # torch.manual_seed(self.config['algorithm']['seed'])
        # tf.random.set_seed(self.config['algorithm']['seed'])

        self.agent = agent_class(self.config['algorithm'], logger_creator=lambda config: UnifiedLogger(config, self.experiment_folder, loggers=None))

        # Step 9: Model & Config Data
        wandb.config.update({'algorithm_final': self.agent.config})
        with open(f'{self.config_folder}agent_config_final.json', "wt") as f:
            pprint.pprint(self.agent.config, stream=f)
        wandb.save(f'{self.config_folder}agent_config_final.json')

        self.analyze_models()


    def analyze_models(self):
        policy = self.agent.get_policy()
        keras_models = {}
        trainable_variables = []

        try:
            keras_models['policy.model.flatten[0].base_model'] = policy.model.flatten[0].base_model
        except AttributeError:
            pass
        try:
            keras_models['policy.model.post_fc_stack.base_model'] = policy.model.post_fc_stack.base_model
        except AttributeError:
            pass
        try:
            keras_models['policy.model.logits_and_value_model'] = policy.model.logits_and_value_model
        except AttributeError:
            pass
        try:
            keras_models['policy.model.base_model'] = policy.model.base_model
        except AttributeError:
            pass
        try:
            keras_models['policy.model.q_value_head'] = policy.model.q_value_head
        except AttributeError:
            pass
        try:
            keras_models['policy.model.state_value_head'] = policy.model.state_value_head
        except AttributeError:
            pass

        for name, model in keras_models.items():
            print(bcolors.OKGREEN, name, bcolors.ENDC)
            model.summary()
            trainable_variables.append(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]))

        print(trainable_variables)
        wandb.config.update({'trainable_variables': sum(trainable_variables)})
        print(f'total trainable parameters: {sum(trainable_variables)}')


    def run(self, epochs=100, silent=False):
        print(f'Starting training of {epochs} epochs @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}')

        try:
            for epoch in range(1, epochs + 1):
                # Step 1: Train epoch
                train_start = time.time()
                result = self.agent.train()
                self.results.append(result)
                self.train_times.append(time.time() - train_start)

                # Step 2: Save checkpoint
                Utils.ensure_storage_connection()
                os.makedirs(self.checkpoints_folder, exist_ok=True)
                self.agent.save(self.checkpoints_folder)

                # Step 3: Print results
                if self.verbose and not silent:
                    self.print_result(result, epoch, epochs)
        except:
            wandb.finish()
            raise

        wandb.finish()
        return self


    def print_result(self, result, epoch, epochs):
        print(f'--------- Epoch {epoch} (Total Samples: {result["info"]["num_env_steps_trained"]}) ---------')

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
        print(f'Iteration Time: {train_time/60:.2f}min ({epochs * (train_time) / 60:.1f}min for {epochs} epochs, {(epochs - epoch) * (train_time) / 60:.1f}min to go)')
        # pprint(result["sampler_perf"])
        # print(f'total time per step: {sum(result["sampler_perf"].values()):.2f}ms')