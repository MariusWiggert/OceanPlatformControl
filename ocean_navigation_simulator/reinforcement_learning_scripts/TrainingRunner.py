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
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelCatalog
from ray.train import Trainer
from ray.tune.logger import UnifiedLogger
import wandb
import pprint
import tensorflow as tf
import torch
import torchinfo
import torchsummary

from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.reinforcement_learning.OceanDenseTorchModel import OceanDenseTorchModel
from ocean_navigation_simulator.reinforcement_learning.OceanDenseTFModel import OceanDenseTFModel
from ocean_navigation_simulator.reinforcement_learning.OceanEnvFactory import OceanEnvFactory
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils
from ocean_navigation_simulator.utils.bcolors import bcolors

"""
    RLRunner takes a configuration to run a RL training with Rllib and hides away all the ugly stuff
    needed to run Rllib. It creates folder for results, saves configurations, registers Environment and
    Model with Rllib and finally prints results after each training iteration.
"""
class TrainingRunner:
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
        self.train_times = []

        # Step 1: Prepare Paths & Folders
        self.timestring = datetime.datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y_%m_%d_%H_%M_%S")
        self.config['folders'] = {}
        self.config['folders']['experiment'] = f"{self.config['experiments_folder']}{self.name}_{self.timestring}/"
        self.config['folders']['config'] = f"{self.config['folders']['experiment']}config/"
        self.config['folders']['source'] = f"{self.config['folders']['experiment']}source/"
        self.config['folders']['results'] = f"{self.config['folders']['experiment']}results/"
        self.config['folders']['checkpoints'] = f"{self.config['folders']['experiment']}checkpoints/"
        Utils.ensure_storage_connection()
        Utils.clean_results(self.config['experiments_folder'], verbose=1, iteration_limit=10, delete=True)
        os.makedirs(self.config['folders']['experiment'])
        os.makedirs(self.config['folders']['config'])
        os.makedirs(self.config['folders']['source'])
        os.makedirs(self.config['folders']['results'])
        os.makedirs(self.config['folders']['checkpoints'])

        # Step 2: Start Weights & Biases
        Utils.ensure_storage_connection()
        wandb.tensorboard.patch(
            root_logdir=self.config['folders']['experiment'],
            tensorboard_x=True
        )
        wandb.init(
            project="RL for underactuated navigation",
            entity="ocean-platform-control",
            dir="/seaweed-storage/",
            name=f'{self.name}_{self.timestring}',
            tags=["baseline" if self.config['environment']['fake'] else 'experiment'] + tags,
            config=self.config,
        )
        with open(f"{self.config['folders']['experiment']}wandb_run_id", 'wt') as f:
            f.write(wandb.run.id)

        # Step 3: Save configuration & source
        json.dump(self.config, open(f"{self.config['folders']['config']}config.json", "w"), indent=4)
        wandb.save(f"{self.config['folders']['config']}config.json")

        # with zipfile.ZipFile(f'{self.experiment_folder}source.zip', 'w') as file:
        #     file.write('config', 'config')
        #     file.write('ocean_navigation_simulation', 'ocean_navigation_simulation')
        #     file.write('scripts', 'scripts')
        #     file.write('setup', 'setup')

        # Step 4: Fix Seeds
        np.random.seed(self.config['algorithm']['seed'])
        random.seed(self.config['algorithm']['seed'])
        torch.manual_seed(self.config['algorithm']['seed'])

        # Step 5: Register Env
        ray.tune.registry.register_env("OceanEnv", OceanEnvFactory(
            config=self.config['environment'],
            feature_constructor_config=self.config['feature_constructor'],
            reward_function_config=self.config['reward_function'],
            folders=self.config['folders'],
            empty_env=False,
            verbose=self.verbose-1,
        ))

        # Step 6: Register Model
        if self.config['algorithm']['model'].get('custom_model', '') == 'OceanDenseTFModel':
            ModelCatalog.register_custom_model("OceanDenseTFModel", OceanDenseTFModel)
        elif self.config['algorithm']['model'].get('custom_model', '') == 'OceanDenseTorchModel':
            ModelCatalog.register_custom_model("OceanDenseTorchModel", OceanDenseTorchModel)

        # Step 7: Custom Metric
        # https://docs.ray.io/en/latest/rllib/rllib-training.html#callbacks-and-custom-metrics
        # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        class CustomCallback(ray.rllib.algorithms.callbacks.DefaultCallbacks):
            def __init__(self):
                super().__init__()
                self.episodes_sampled = 0
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
                episode.custom_metrics["episode_length"] = info["average_step_time"]
                # episode.custom_metrics["problem_index"] = info["problem"].extra_info['index']
                self.episodes_sampled += 1
            def on_train_result(self,*,result: dict, algorithm: Optional["Algorithm"] = None,trainer=None, **kwargs):
                for k, v in result['custom_metrics'].copy().items():
                    if len(v) > 0:
                        result['custom_metrics'][f"{k}_min"] = min(v)
                        result['custom_metrics'][f"{k}_mean"] = sum(v) / len(v)
                        result['custom_metrics'][f"{k}_max"] = max(v)
                result['custom_metrics']['episodes_sampled'] = self.episodes_sampled
                self.episodes_sampled = 0
        self.config['algorithm']["callbacks"] = CustomCallback

        # Step 7: Create Agent
        if self.config['algorithm_name'] == 'apex-dqn':
            trainer_class = ApexDQN
        else:
            raise ValueError(f"Algorithm '{self.config['algorithm_name']}' not implemented.")
        self.trainer = trainer_class(self.config['algorithm'], logger_creator=lambda config: UnifiedLogger(config, self.config['folders']['experiment'], loggers=None))

        # Step 9: Model & Config Data
        wandb.config.update({'algorithm_final': self.trainer.config})
        with open(f"{self.config['folders']['config']}agent_config_final.json", "wt") as f:
            pprint.pprint(self.trainer.config, stream=f)
        wandb.save(f"{self.config['folders']['config']}agent_config_final.json")

        if self.config['algorithm']['framework'] == 'torch':
            self.analyze_models_torch()
        else:
            self.analyze_models_tf()

    def analyze_models_torch(self):
        policy = self.trainer.get_policy()
        print('Policy Class', policy)

        torch_models = {}
        trainable_variables = []

        try:
            torch_models['policy.model._hidden_layers'] = (policy.model._hidden_layers, policy.model.obs_space.shape)
        except AttributeError:
            pass
        try:
            torch_models['policy.model._value_branch'] = (policy.model._value_branch, (policy.model.num_outputs,))
        except AttributeError:
            pass
        try:
            torch_models['policy.model.advantage_module'] = (policy.model.advantage_module, (policy.model.num_outputs,))
        except AttributeError:
            pass
        try:
            torch_models['policy.model.value_module'] = (policy.model.value_module, (policy.model.num_outputs,))
        except AttributeError:
            pass

        for name, (model, shape) in torch_models.items():
            print(bcolors.OKGREEN, name, bcolors.ENDC)
            torchsummary.summary(model, input_size=shape)
            trainable_variables.append([p.numel() for p in model.parameters() if p.requires_grad])

        total_trainable_variables = sum([sum(t) for t in trainable_variables])
        print(f'trainable parameters: {trainable_variables}')
        print(f'total trainable parameters: {total_trainable_variables}')
        wandb.config.update({'trainable_variables_detail': trainable_variables})
        wandb.config.update({'trainable_variables': total_trainable_variables})

    def analyze_models_tf(self):
        policy = self.trainer.get_policy()
        print('Policy Class', policy)

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
            trainable_variables.append([np.prod(v.get_shape().as_list()).item() for v in model.trainable_variables])

        total_trainable_variables = sum([sum(t) for t in trainable_variables])
        print(f'trainable parameters: {trainable_variables}')
        print(f'total trainable parameters: {total_trainable_variables}')
        wandb.config.update({'trainable_variables_detail': trainable_variables})
        wandb.config.update({'trainable_variables': total_trainable_variables})

    def run(self, epochs=100, silent=False):
        print(f'Starting training of {epochs} epochs @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}')

        try:
            for epoch in range(1, epochs + 1):
                # Step 1: Train epoch
                train_start = time.time()
                result = self.trainer.train()
                self.results.append(result)
                self.train_times.append(time.time() - train_start)

                with open(f"{self.config['folders']['results']}epoch{epoch}.json", "wt") as f:
                    pprint.pprint(result, stream=f)

                # Step 2: Save checkpoint
                Utils.ensure_storage_connection()
                self.trainer.save(checkpoint_dir=self.config['folders']['checkpoints'])
                # self.trainer.get_policy().export_model(export_dir=f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/")
                # self.trainer.get_policy().export_model(export_dir=f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/", onnx=15)

                # Step 3: Print results
                if self.verbose and not silent:
                    self.print_result(result, epoch, epochs)

                # wandb.synch()
        except:
            wandb.finish()
            raise

        wandb.finish()
        return self

    def print_result(self, result, epoch, epochs):
        print(f'--------- Epoch {epoch} ---------')

        print(f'-- Custom Metrics --')
        for k, v in result['custom_metrics'].items():
            if 'mean' in k:
                print(f'{k}: {v:.2f}')
        print(' ')

        print(f'-- Episode Rewards [Min: {result["episode_reward_min"]:.1f}, Mean: {result["episode_reward_mean"]:.2f}, Max: {result["episode_reward_max"]:.1f}]  --')
        print(f'[{", ".join([f"{elem:.1f}" for elem in result["hist_stats"]["episode_reward"][-min(50, result["episodes_this_iter"]):]])}]')
        print(' ')

        episodes_this_iteration = result["hist_stats"]["episode_lengths"][-result["episodes_this_iter"]:]
        print(f'-- Reported Episode Length [Min: {min(episodes_this_iteration):.0f}, Mean: {result["episode_len_mean"]:.1f}, Max: {max(episodes_this_iteration):.0f}] --')
        print(result["hist_stats"]["episode_lengths"][-min(50, result["episodes_this_iter"]):])
        print(f'Episodes Sampled: {result["episodes_this_iter"]:,} (Total: {result["episodes_total"]:,}, custom: {result["custom_metrics"]["episodes_sampled"]:,})')
        print(f'Episode Steps Sampled: {sum(episodes_this_iteration):,} (Total: {sum(result["hist_stats"]["episode_lengths"]):,})')
        print(' ')

        print(f'-- Enivironment Steps --')
        print(f'Env Steps Sampled:  {result["num_env_steps_sampled_this_iter"]:,} (Total: {result["num_env_steps_sampled"]:,})')
        print(f'Env Steps Trained:  {result["num_env_steps_trained_this_iter"]:,} (Total: {result["num_env_steps_trained"]:,})')
        print(' ')

        print(f'-- Average Step Time: {sum(result["sampler_perf"].values()):.2f}ms --')
        print(result["sampler_perf"])
        print(' ')

        print('-- Timing --')
        print(f'Iteration Time: {self.train_times[-1]/60:.2f}min ({(epochs * (self.train_times[-1])) // 3600}h {((epochs * (self.train_times[-1])) % 3600) / 60:.1f}min for {epochs} epochs, {(epochs - epoch) * (self.train_times[-1]) / 60:.1f}min to go)')