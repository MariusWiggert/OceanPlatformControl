import datetime
import logging
import os
import pickle
import pprint
import shutil
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import ray
from matplotlib import pyplot as plt

from ocean_navigation_simulator.reinforcement_learning.scripts.MissionGenerator import (
    MissionGenerator,
)
from ocean_navigation_simulator.utils import cluster_utils
from ocean_navigation_simulator.utils.misc import get_process_information_dict


class GenerationRunner:
    """
    The GenerationRunner generates missions with cached hindcast & forecast planners
     - this greatly speeds up the training by using the cached hj planners
     - each batch generates 1 target and samples various starts (usually 4 or 8)
     - batches are arbitrary grouped to decrease the amount of folders (to open up in sftp)
     - each batch is run in a separate process and retried several times to handle memory overflow of hj planner
    """

    def __init__(self, name: str, config: {}, verbose: Optional[int] = 0):
        self.name = name
        self.config = config
        self.verbose = verbose

        print(
            f'GenerationRunner: Generating {config["size"]["groups"] * config["size"]["batches_per_group"]} batches in {config["size"]["groups"]} groups'
        )

        # Step 1: Prepare Paths & Save configuration
        self.timestring = datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime(
            "%Y_%m_%d_%H_%M_%S"
        )
        self.results_folder = f'{config["generation_folder"]}/{self.name}_{self.timestring}/'
        self.config["results_folder"] = self.results_folder
        cluster_utils.ensure_storage_connection()
        os.makedirs(self.results_folder)
        os.makedirs(f"{self.results_folder}config/")
        with open(f"{self.results_folder}config/config.pickle", "wb") as f:
            pickle.dump(self.config, f)
        with open(f"{self.results_folder}config/config.json", "wt") as f:
            pprint.pprint(self.config, stream=f)

        # Step 2: Run Generation with Ray
        self.ray_results = ray.get(
            [
                self.generate_batch_ray.options(
                    num_cpus=config["ray_options"]["resources"]["CPU"],
                    num_gpus=config["ray_options"]["resources"]["GPU"],
                    max_retries=config["ray_options"]["max_retries"],
                    resources={
                        i: config["ray_options"]["resources"][i]
                        for i in config["ray_options"]["resources"]
                        if i != "CPU" and i != "GPU"
                    },
                ).remote(
                    results_folder=self.results_folder,
                    scenario_file=config["scenario_file"],
                    mission_generation_config=config["mission_generation"],
                    group=int(batch // config["size"]["batches_per_group"]),
                    batch=batch,
                    batch_size=config["size"]["batch_size"],
                    verbose=verbose,
                )
                for batch in range(config["size"]["groups"] * config["size"]["batches_per_group"])
            ]
        )
        self.problems = [problem for problems in self.ray_results for problem in problems]

        # Step 3: Save Results
        cluster_utils.ensure_storage_connection()
        self.problems_df = pd.DataFrame(self.problems)
        self.problems_df.to_csv(f"{self.results_folder}problems.csv")

        # Step 4: Analyse Generation
        GenerationRunner.analyse_generation(self.results_folder)
        GenerationRunner.plot_generation(self.results_folder)

    # Memory Leak of HJ Planner: Only run on workers and with a new process each
    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-remote
    @staticmethod
    @ray.remote(max_calls=1)
    def generate_batch_ray(
        results_folder: str,
        scenario_file: str,
        mission_generation_config: dict,
        group: int,
        batch: int,
        batch_size: int,
        verbose,
    ):
        # Supress TF CPU warnings:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
        logging.getLogger("tensorflow").setLevel(logging.FATAL)
        logging.getLogger("absl").setLevel(logging.FATAL)

        try:
            batch_start_time = time.time()
            batch_folder = f"{results_folder}groups/group_{group}/batch_{batch}/"

            if verbose > 0:
                print(f"GenerationRunner: Starting Batch {batch} Group {group}")

            # Step 1: Generate Batch
            generate_start = time.time()
            problem_factory = MissionGenerator(
                scenario_file=scenario_file,
                config={"seed": batch} | mission_generation_config,
                verbose=verbose - 1,
            )
            problems = problem_factory.generate_batch(batch_size)
            generate_time = time.time() - generate_start

            # Step 2: Plot Batch
            if mission_generation_config["plot_batch"]:
                plot_start = time.time()
                cluster_utils.ensure_storage_connection()
                os.makedirs(batch_folder, exist_ok=True)
                problem_factory.plot_batch(batch_size, filename=f"{batch_folder}animation.gif")
                plot_time = time.time() - plot_start
            else:
                plot_time = 0

            # step 3: Save Hindcast Planner
            cluster_utils.ensure_storage_connection()
            problem_factory.hindcast_planner.save_planner_state(f"{batch_folder}hindcast_planner/")

            # Step 4: Run & Save Forecast Planner
            forecast_start = time.time()
            problem_factory.run_forecast(batch_folder=batch_folder)
            forecast_time = time.time() - forecast_start

            # Step 5: Format & Save Batch Results
            cluster_utils.ensure_storage_connection()
            os.makedirs(batch_folder, exist_ok=True)
            batch_info = {
                "group": group,
                "batch": batch,
                "total_time": f"{ time.time()-batch_start_time:.2f}s",
                "generate_time": generate_time,
                "plot_time": plot_time,
                "forecast_time": forecast_time,
            } | get_process_information_dict()
            batch_results = [
                problem.to_dict() | batch_info for index, problem in enumerate(problems)
            ]
            pd.DataFrame(batch_results).to_csv(f"{batch_folder}problems.csv")

            if verbose > 0:
                print(
                    f'GenerationRunner: Finished Batch {batch} Group {group} (total: {batch_info["total_time"]:.1f}s, generate: {generate_time:.1f}s, plotting: {plot_time:.1f}s, forecast: {forecast_time:.1f}s), RAM: {batch_info["process_ram"]}'
                )

        except Exception as e:
            shutil.rmtree(batch_folder, ignore_errors=True)
            if verbose > 0:
                print(f"GenerationRunner: Aborted Batch {batch} Group {group}")
                print(e)
            sys.exit()

        return batch_results

    @staticmethod
    def analyse_generation(
        results_folder: str,
    ):
        cluster_utils.ensure_storage_connection()
        problems_df = pd.read_csv(f"{results_folder}problems.csv")

        ram_numerical = np.array(
            [
                float(row["batch_ram"].removesuffix("MB").replace(",", ""))
                for index, row in problems_df.iterrows()
            ]
        )
        print(f"RAM Min:  {ram_numerical.min():.1f}")
        print(f"RAM Mean: {ram_numerical.mean():.1f}")
        print(f"RAM Max:  {ram_numerical.max():.1f}")

        time_numerical = np.array(
            [float(row["batch_time"].removesuffix("s")) for index, row in problems_df.iterrows()]
        )
        print(f"Batch Time Min:  {time_numerical.min():.1f}s")
        print(f"Batch Time Mean: {time_numerical.mean():.1f}s")
        print(f"Batch Time Max:  {time_numerical.max():.1f}s")

    @staticmethod
    def plot_generation(
        results_folder: str,
    ):
        # Step 1:
        cluster_utils.ensure_storage_connection()
        problems_df = pd.read_csv(f"{results_folder}problems.csv")
        target_df = problems_df[problems_df["factory_index"] == 0]
        analysis_folder = f"{results_folder}analysis/"
        os.makedirs(analysis_folder, exist_ok=True)

        # Step 2:
        plt.figure(figsize=(12, 12))
        plt.scatter(
            target_df["x_T_lon"], target_df["x_T_lat"], c="green", marker="x", label="target"
        )
        plt.scatter(
            problems_df["x_0_lon"], problems_df["x_0_lat"], c="red", marker="o", label="start"
        )
        plt.savefig(f"{analysis_folder}starts_and_targets.png")
        plt.show()