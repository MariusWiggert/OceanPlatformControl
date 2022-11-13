import datetime
import json
import os
import pprint
import random
import shutil
import time
from typing import Dict, List, Optional

import gym
import numpy as np
import pytz
import ray
import torch
import wandb
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
            self.config["experiments_folder"], verbose=1, iteration_limit=10, delete=False
        )
        self.timestring = datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime(
            "%Y_%m_%d_%H_%M_%S"
        )
        experiment_path = f"{self.config['experiments_folder']}{self.name}_{self.timestring}/"
        self.config["folders"] = {
            "experiment": experiment_path,
            "config": f"{experiment_path}config/",
            "source": f"{experiment_path}source/",
            "results": f"{experiment_path}results/",
            "checkpoints": f"{experiment_path}checkpoints/",
        }
        for f in self.config["folders"].values():
            os.makedirs(f)

        # Step 2: Start Weights & Biases
        cluster_utils.ensure_storage_connection()
        wandb.tensorboard.patch(
            root_logdir=self.config["folders"]["experiment"], tensorboard_x=True
        )
        wandb.init(
            project="seaweed-rl",
            entity="jeromejeannin",
            dir="/seaweed-storage/",
            name=f"{self.name}_{self.timestring}",
            tags=["baseline" if self.config["environment"]["fake"] else "experiment"] + tags,
            config=self.config,
        )
        with open(f"{self.config['folders']['experiment']}wandb_run_id", "wt") as f:
            f.write(wandb.run.id)

        # Step 3: Save configuration & source
        json.dump(
            self.config, open(f"{self.config['folders']['config']}config.json", "w"), indent=4
        )
        wandb.save(f"{self.config['folders']['config']}config.json")
        # with zipfile.ZipFile(f'{self.experiment_folder}source.zip', 'w') as file:
        #     file.write('config', 'config')
        #     file.write('ocean_navigation_simulation', 'ocean_navigation_simulation')
        #     file.write('scripts', 'scripts')
        #     file.write('setup', 'setup')

        # Step 4: Fix Seeds
        np.random.seed(self.config["algorithm"]["seed"])
        random.seed(self.config["algorithm"]["seed"])
        torch.manual_seed(self.config["algorithm"]["seed"])

        # Step 5: Custom Metric
        # https://docs.ray.io/en/latest/rllib/rllib-training.html#callbacks-and-custom-metrics
        # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        class CustomCallback(ray.rllib.algorithms.callbacks.DefaultCallbacks):
            def __init__(self):
                super().__init__()
                self.episodes_sampled = 0

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
                if info["problem_status"] != 0:
                    episode.custom_metrics["success"] = info["problem_status"] > 0
                if info["problem_status"] > 0:
                    episode.custom_metrics["arrival_time_in_h"] = info["arrival_time_in_h"]
                episode.custom_metrics["problem_status"] = info["problem_status"]
                episode.custom_metrics["ram_usage_MB"] = info["ram_usage_MB"]
                episode.custom_metrics["episode_time"] = info["episode_time"]
                episode.custom_metrics["average_step_time"] = info["average_step_time"]
                episode.custom_metrics["episode_length"] = info["average_step_time"]
                # episode.custom_metrics["problem_index"] = info["problem"].extra_info['index']
                self.episodes_sampled += 1

            def on_train_result(
                self,
                *,
                result: dict,
                algorithm: Optional[Algorithm] = None,
                trainer=None,
                **kwargs,
            ):
                for k, v in result["custom_metrics"].copy().items():
                    if len(v) > 0:
                        result["custom_metrics"][f"{k}_min"] = min(v)
                        result["custom_metrics"][f"{k}_mean"] = sum(v) / len(v)
                        result["custom_metrics"][f"{k}_max"] = max(v)
                result["custom_metrics"]["episodes_sampled"] = self.episodes_sampled
                self.episodes_sampled = 0

        self.config["algorithm"]["callbacks"] = CustomCallback

        # Step 6: Create Trainer
        self.trainer = TrainerFactory.create(
            config=self.config,
            logger_path=self.config["folders"]["experiment"],
            verbose=verbose - 1,
        )

        # Step 7: Model & Config Data
        wandb.config.update({"algorithm_final": self.trainer.config})
        with open(f"{self.config['folders']['config']}agent_config_final.json", "wt") as f:
            pprint.pprint(self.trainer.config, stream=f)
        wandb.save(f"{self.config['folders']['config']}agent_config_final.json")

        self.analyze_custom_torch_model()

    def analyze_custom_torch_model(self):
        policy = self.trainer.get_policy()

        print("Policy Class", policy)
        print("Preprocessor", self.trainer.workers.local_worker().preprocessors)
        print("Filter", self.trainer.workers.local_worker().filters)

        # Model Informations
        self.model = policy.model
        if isinstance(self.model.obs_space, gym.spaces.Tuple):
            shape = [o.shape for o in self.model.obs_space]
            self.dummy_input = [torch.randn((32,) + s) for s in shape]
        else:
            shape = self.model.obs_space.shape
            self.dummy_input = torch.randn((32,) + shape)
        trainable_variables = [p.numel() for p in self.model.parameters() if p.requires_grad]
        trainable_variables_modules = [
            sum([p.numel() for p in m.parameters() if p.requires_grad])
            for m in self.model.children()
        ]
        trainable_variables_modules_named = {
            n: sum([p.numel() for p in m.parameters() if p.requires_grad])
            for n, m in self.model.named_children()
        }
        total_trainable_variables = sum(trainable_variables)
        print(f"trainable parameters: {trainable_variables}")
        print(f"trainable parameters per module: {trainable_variables_modules}")
        print(f"trainable parameters per module: {trainable_variables_modules_named}")
        print(f"total trainable parameters: {total_trainable_variables}")

        # Export Model
        epoch = 0
        cluster_utils.ensure_storage_connection()
        os.makedirs(f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/")
        torch.save(
            self.model, f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/model.pt"
        )
        if isinstance(self.dummy_input, list):
            torch.onnx.export(
                self.model,
                args=tuple(i.cuda() for i in self.dummy_input),
                f=f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/model.onnx",
                export_params=True,
                opset_version=15,
            )
        else:
            torch.onnx.export(
                self.model,
                self.dummy_input.cuda(),
                f=f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/model.onnx",
                export_params=True,
                opset_version=15,
            )

        # Model Summary
        model_summary = torchinfo.summary(
            self.model, input_data=self.dummy_input, depth=10, verbose=1
        )
        with open(
            f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/model.log", "wt"
        ) as f:
            f.write(model_summary.__repr__())
        wandb.save(f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/model.log")

        # Weights & Biases
        wandb.watch(
            models=self.model,
            log="all",
            log_freq=1,
        )
        wandb.config.update({"trainable_variables_detail": trainable_variables})
        wandb.config.update({"trainable_variables_module": trainable_variables_modules})
        wandb.config.update({"trainable_variables_module_named": trainable_variables_modules_named})
        wandb.config.update({"trainable_variables": total_trainable_variables})

    def run(self, epochs=100, silent=False):
        print(
            f'Starting training of {epochs} epochs @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}'
        )

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
                cluster_utils.ensure_storage_connection()
                self.trainer.save(checkpoint_dir=self.config["folders"]["checkpoints"])
                # torch.onnx.export(self.model, args=(self.dummy_input[0].cuda(), self.dummy_input[1].cuda()), f=f"{self.config['folders']['checkpoints']}checkpoint_{epoch:06d}/model.onnx", export_params=True, opset_version=15)

                # Step 3: Print results
                if self.verbose and not silent:
                    self.print_result(result, epoch, epochs)

                # wandb.synch()
        except KeyboardInterrupt:
            wandb.finish()
            raise

        wandb.finish()
        return self

    def print_result(self, result, epoch, epochs):
        print(f"--------- Epoch {epoch} ---------")

        print("-- Custom Metrics --")
        for k, v in result["custom_metrics"].items():
            if "mean" in k:
                print(f"{k}: {v:.2f}")
        print(" ")

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
            f'Episodes Sampled: {result["episodes_this_iter"]:,} (Total: {result["episodes_total"]:,}, custom: {result["custom_metrics"]["episodes_sampled"]:,})'
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
            f"Iteration Time: {self.train_times[-1] / 60:.2f}min ({(epochs * (self.train_times[-1])) // 3600}h {((epochs * (self.train_times[-1])) % 3600) / 60:.1f}min for {epochs} epochs, {(epochs - epoch) * (self.train_times[-1]) / 60:.1f}min to go)"
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
