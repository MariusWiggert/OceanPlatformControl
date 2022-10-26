import datetime
import json
import os
import pprint
import random
import shutil
import time
from functools import partial
from typing import Dict, List, Optional
from statistics import mean, median

import gym
import numpy as np
import pytz
import ray
import torch
import wandb
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation import Episode

from torchinfo import torchinfo

from ocean_navigation_simulator.reinforcement_learning.TrainerFactory import (
    TrainerFactory,
)
from ocean_navigation_simulator.utils import cluster_utils
from ocean_navigation_simulator.utils.misc import bcolors


class TrainingRunner:
    """
    RLRunner takes a configuration to run a RL training with Rllib:
    - hides away all the ugly stuff needed to run Rllib (registers Environment and Model)
    - creates folder for results
    - saves configurations
    - prints results after each training iteration.
    """

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
        cluster_utils.ensure_storage_connection()
        TrainingRunner.clean_results(
            self.config["experiments_folder"], verbose=0, iteration_limit=10, delete=True
        )
        self.timestring = datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime(
            "%Y_%m_%d_%H_%M_%S"
        )
        experiment_path = f"{self.config['experiments_folder']}{self.name}_{self.timestring}/"
        self.config["folders"] = {
            "experiment": experiment_path,
            "config": f"{experiment_path}config/",
            "results": f"{experiment_path}results/",
            "checkpoints": f"{experiment_path}checkpoints/",
        }
        for f in self.config["folders"].values():
            os.makedirs(f)

        # Step 2: Start Weights & Biases
        cluster_utils.ensure_storage_connection()
        wandb.init(
            project="seaweed-rl",
            entity="jeromejeannin",
            dir="/tmp/",
            name=f"{self.name}_{self.timestring}",
            tags=["baseline" if self.config["environment"]["fake"] not in [False, 'residual'] else "experiment"] + tags,
            config=self.config,
        )
        with open(f"{self.config['folders']['experiment']}wandb_run_id", "wt") as f:
            f.write(wandb.run.id)

        # Step 3: Save configuration
        with open(f"{self.config['folders']['config']}config.json", "w") as f:
            json.dump(self.config, f, indent=4)

        # Step 4: Custom Metric
        # https://docs.ray.io/en/latest/rllib/rllib-training.html#callbacks-and-custom-metrics
        # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        class CustomCallback(ray.rllib.algorithms.callbacks.DefaultCallbacks):
            def __init__(self):
                super().__init__()

            def on_episode_end(
                self,
                *,
                worker: RolloutWorker,
                base_env: BaseEnv,
                policies: Dict[str, Policy],
                episode: Episode,
                env_index: int,
            ):
                info = episode.last_info_for()
                for k, v in reduce_dict(info).items():
                    episode.custom_metrics['hist_stats/'+k] = v

            def on_train_result(
                self,
                *,
                result: dict,
                algorithm: Optional[Algorithm] = None,
                trainer=None,
                **kwargs,
            ):
                def add_statistics(metrics):
                    for k, v in metrics.copy().items():
                        k = k.replace('hist_stats/', '')
                        metrics[f"{k}_min"] = min(v)
                        metrics[f"{k}_mean"] = mean(v)
                        metrics[f"{k}_median"] = median(v)
                        metrics[f"{k}_max"] = max(v)
                    return metrics

                for k, v in add_statistics(result["custom_metrics"]).items():
                    result["custom_metrics"][k] = v
                if "evaluation" in result:
                    for k, v in add_statistics(result["evaluation"]["custom_metrics"]).items():
                        result["evaluation"]["custom_metrics"][k] = v


        self.config["algorithm"]["callbacks"] = CustomCallback

        # Step 5: Create Trainer
        self.trainer = TrainerFactory.create(
            config=self.config,
            logger_path=self.config["folders"]["experiment"],
            verbose=verbose - 1,
        )

        # Step 6: Model & Config Data
        wandb.config.update({"algorithm_final": self.trainer.config})
        with open(f"{self.config['folders']['config']}agent_config_final.json", "wt") as f:
            pprint.pprint(self.trainer.config, stream=f)

        self.analyze_model()

    def analyze_model(self):
        self.policy = self.trainer.get_policy()
        self.model = self.policy.model

        policy_info = {
            "Policy Class": str(self.policy),
            "Preprocessor": str(self.trainer.workers.local_worker().preprocessors),
            "Filter": str(self.trainer.workers.local_worker().filters),
        }
        print("Policy Info:", policy_info)
        wandb.run.summary.update(reduce_dict({'Policy Info': policy_info}))

        # Trainable Variables
        layers_variables = [p.numel() for p in self.model.parameters() if p.requires_grad]
        module_variables = {
            n: sum([p.numel() for p in m.parameters() if p.requires_grad])
            for n, m in self.model.named_children()
        }
        model_variables = {
            "Total": sum(layers_variables),
            "Layers": layers_variables,
            "Modules": list(module_variables.values()),
            "Modules Named": module_variables,
        }
        print("Model Variables:", model_variables)
        wandb.run.summary.update(reduce_dict({"Model Variables": model_variables}))

        # Model Informations
        if isinstance(self.model.obs_space, gym.spaces.Tuple):
            shape = [o.shape for o in self.model.obs_space]
            self.dummy_input = [torch.randn((32,) + s) for s in shape]
        else:
            shape = self.model.obs_space.shape
            self.dummy_input = torch.randn((32,) + shape)

        # Export Model
        epoch = 0
        self.export_model(epoch)

        # Model Summary
        model_info = torchinfo.summary(
            self.model,
            input_data=self.dummy_input,
            depth=10,
            verbose=1,
            col_names=["input_size","output_size","num_params","kernel_size","trainable"],
            row_settings=("depth", "var_names"),
        ).__repr__()
        with open(
            f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/model.log", "wt"
        ) as f:
            f.write(model_info)

    def export_model(self, epoch):
        cluster_utils.ensure_storage_connection()
        folder = f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/"

        os.makedirs(folder, exist_ok=True)

        torch.save(self.model, folder + "model.pt")
        if isinstance(self.dummy_input, list):
            torch.onnx.export(
                self.model,
                args=tuple(i.cuda() for i in self.dummy_input),
                f=folder + "model.onnx",
                export_params=True,
                opset_version=15,
            )
        else:
            torch.onnx.export(
                self.model,
                self.dummy_input.cuda(),
                f=folder + "model.onnx",
                export_params=True,
                opset_version=15,
            )

    def run(self, epochs=100, silent=False, checkpoint_freq=1):
        print(
            f'Starting training of {epochs} epochs @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}'
        )

        wandb.watch(
            models=self.trainer.get_policy().model,
            log='all',
            log_freq=1,
        )

        try:
            for epoch in range(1, epochs + 1):
                # self.gradients = {}
                #
                # def hook(name, grad):
                #     if name in self.gradients:
                #         self.gradients[name] = torch.cat((self.gradients[name], grad.data.reshape(-1)),0)
                #     else:
                #         self.gradients[name] = grad.data.reshape(-1)
                #
                # for name, p in self.trainer.get_policy().model.named_parameters():
                #     if p.requires_grad:
                #         p.register_hook(partial(hook, name))

                # Step 1: Train
                train_start = time.time()
                rllib_result = self.trainer.train()
                self.train_times.append(time.time() - train_start)

                rllib_result.pop('config', None)
                weights = {n: p.data for n, p in self.trainer.get_policy().model.named_parameters() if p.requires_grad}
                results = reduce_dict({
                    "ray/tune": rllib_result,
                    # "gradients": self.gradients,
                    "weights": weights,
                })
                for k, v in results.copy().items():
                    if isinstance(v, list):
                        results[k] = np.array(v)
                    if isinstance(v, str):
                        del results[k]
                    if isinstance(v, bool):
                        results[k] = int(v)

                # pprint.pprint(results)
                self.results.append(results)
                wandb.log(results, step=epoch, commit=True)

                cluster_utils.ensure_storage_connection()
                # with open(f"{self.config['folders']['results']}epoch{epoch}.json", "wt") as f:
                #     json.dump(result, f, indent=4)

                # Step 2: Save checkpoint
                if epoch % checkpoint_freq == 0:
                    cluster_utils.ensure_storage_connection()
                    self.trainer.save(checkpoint_dir=self.config["folders"]["checkpoints"])
                    # self.export_model(epoch)

                # Step 3: Print results
                if self.verbose and not silent:
                    self.print_result(rllib_result, epoch, epochs)

                # wandb.synch()
        except KeyboardInterrupt:
            wandb.finish()
            raise

        wandb.finish()
        return self

    def print_result(self, result, epoch, epochs):
        print(f"--------- Epoch {epoch} ---------")

        if result["episodes_this_iter"] > 0:

            def print_custom_metrics(result, eval=False):
                print(f"-- Custom Metrics {'Evaluation' if eval else ''}--")
                # for k, v in result["custom_metrics"].items():
                #     if "mean" in k:
                #         print(f"{k}: {v:.2f}")
                # print(" ")

            print_custom_metrics(result['sampler_results'])
            if "evaluation" in result['sampler_results']:
                print_custom_metrics(result['sampler_results']["evaluation"], eval=True)

            print(
                f'-- Episode Rewards [Min: {result["episode_reward_min"]:.1f}, Mean: {result["episode_reward_mean"]:.2f}, Max: {result["episode_reward_max"]:.1f}]  --'
            )
            print(
                f'[{", ".join([f"{elem:.1f}" for elem in result["hist_stats"]["episode_reward"][-min(50, result["episodes_this_iter"]):]])}]'
            )
            print(" ")

            episodes_this_iteration = result["hist_stats"]["episode_lengths"][
                -result["episodes_this_iter"] :
            ]
            print(
                f'-- Reported Episode Length [Min: {min(episodes_this_iteration):.0f}, Mean: {result["episode_len_mean"]:.1f}, Max: {max(episodes_this_iteration):.0f}] --'
            )
            print(result["hist_stats"]["episode_lengths"][-min(50, result["episodes_this_iter"]) :])
            print(
                f'Episodes Sampled: {result["episodes_this_iter"]:,} (Total: {result["episodes_total"]:,})'
            )
            print(
                f'Episode Steps Sampled: {sum(episodes_this_iteration):,} (Total: {sum(result["hist_stats"]["episode_lengths"]):,})'
            )
            print(" ")

        print("-- Enivironment Steps --")
        print(
            f'Env Steps Sampled:  {result["num_env_steps_sampled_this_iter"]:,} (Total: {result["num_env_steps_sampled"]:,})'
        )
        print(
            f'Env Steps Trained:  {result["num_env_steps_trained_this_iter"]:,} (Total: {result["num_env_steps_trained"]:,})'
        )
        print(" ")

        print(f'-- Average Step Time: {sum(result["sampler_perf"].values()):.2f}ms --')
        print(result["sampler_perf"])
        print(" ")

        print("-- Timing --")
        print(
            f"Iteration Time: {self.train_times[-1]/60:.2f}min ({(epochs * (self.train_times[-1])) // 3600}h {((epochs * (self.train_times[-1])) % 3600) / 60:.1f}min for {epochs} epochs, {(epochs - epoch) * (self.train_times[-1]) / 60:.1f}min to go)"
        )

    @staticmethod
    def clean_results(
        folder: str = "~/ray_results",
        filter: str = "",
        iteration_limit: Optional[int] = 10,
        delete: Optional[bool] = False,
        ignore_most_recent: Optional[int] = 1,
        verbose: Optional[int] = 0,
    ):
        """
        Ray clogs up the ~/ray_results directory by creating folders for every training start, even when
        canceling after a few iterations. This script removes all short trainings in order to simplify
        finding the important trainings in tensorboard. It however ignores the very last experiment,
        since it could be still ongoing.
        """
        experiments = [
            os.path.join(folder, file)
            for file in os.listdir(folder)
            if not file.startswith(".") and file.startswith(filter)
        ]
        experiments.sort(key=lambda x: os.path.getmtime(x))

        for experiment in (
            experiments[:-ignore_most_recent] if ignore_most_recent > 0 else experiments
        ):
            csv_file = experiment + "/progress.csv"
            if os.path.isfile(csv_file):
                with open(csv_file) as file:
                    row_count = sum(1 for line in file)
                    if row_count < iteration_limit:
                        if delete:
                            shutil.rmtree(experiment, ignore_errors=True)
                        if verbose > 0:
                            print(
                                f"RayUtils.clean_ray_results: Delete {bcolors.FAIL}{experiment} with {row_count} rows {bcolors.ENDC}"
                            )
                    else:
                        if verbose > 0:
                            print(
                                f"RayUtils.clean_ray_results: Keep   {bcolors.OKGREEN}{experiment} with {row_count} rows {bcolors.ENDC}"
                            )
            else:
                if delete:
                    shutil.rmtree(experiment, ignore_errors=True)
                if verbose > 0:
                    print(
                        f"RayUtils.clean_ray_results: Delete {bcolors.FAIL}{experiment} without progress.csv file {bcolors.ENDC}"
                    )


def reduce_dict(ob):
    if type(ob) is dict:
        new = {}
        for k1, v1 in ob.items():
            if type(v1) is dict:
                for k2, v2 in reduce_dict(v1).items():
                    new[k1 + '/' + k2] = v2
            else:
                new[k1] = v1
        return new
    else:
        return dict