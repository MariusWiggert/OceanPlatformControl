import datetime
import logging
import os
import pickle
import pprint
import shutil
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.reinforcement_learning.missions.CachedNavigationProblem import (
    CachedNavigationProblem,
)
from ocean_navigation_simulator.utils import cluster_utils
from ocean_navigation_simulator.utils.misc import (
    bcolors,
    get_process_information_dict,
    set_arena_loggers,
    silence_ray_and_tf,
)

sns.set_theme()


class GenerationRunner:
    """
    The GenerationRunner generates missions with cached hindcast & forecast planners
     - this greatly speeds up the training by using the cached hj planners
     - each batch generates 1 target and samples various starts (usually 4 or 8)
     - batches are arbitrary grouped to decrease the amount of folders (to open up in sftp)
     - each batch is run in a separate process and retried several times to handle memory overflow of hj planner
    """

    def __init__(self, name: str, config: dict, verbose: Optional[int] = 0):
        self.name = name
        self.config = config
        self.verbose = verbose

        # Step 1: Prepare Paths & Save configuration
        self.timestring = datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime(
            "%Y_%m_%d_%H_%M_%S"
        )
        self.results_folder = f'{config["generation_folder"]}/{self.name}_{self.timestring}/'
        self.config["generation"] = self.results_folder
        cluster_utils.ensure_storage_connection()
        os.makedirs(self.results_folder)
        os.makedirs(f"{self.results_folder}config/")
        with open(f"{self.results_folder}config/config.pickle", "wb") as f:
            pickle.dump(self.config, f)
        with open(f"{self.results_folder}config/config.json", "wt") as f:
            pprint.pprint(self.config, stream=f)

        print(
            "GenerationRunner: Generating {b} batches in {g} groups ({f})".format(
                b=config["size"]["groups"] * config["size"]["batches_per_group"],
                g=config["size"]["groups"],
                f=self.results_folder,
            )
        )

        # Step 2: Run Generation with Ray
        self.ray_results = ray.get(
            [
                self.generate_batch_ray.options(
                    num_cpus=config["ray_options"]["resources"].get("CPU", 1.0),
                    num_gpus=config["ray_options"]["resources"].get("GPU", 0.0),
                    max_retries=config["ray_options"].get("max_retries", 10),
                    resources={
                        i: config["ray_options"]["resources"][i]
                        for i in config["ray_options"]["resources"]
                        if i != "CPU" and i != "GPU"
                    },
                ).remote(
                    results_folder=self.results_folder,
                    mission_generation_config=config["mission_generation"],
                    group=int(batch // config["size"]["batches_per_group"]),
                    batch=batch,
                    verbose=verbose,
                )
                for batch in range(config["size"]["groups"] * config["size"]["batches_per_group"])
            ]
        )
        self.problems = [problem for results in self.ray_results for problem in results[0]]
        self.errors = [error for results in self.ray_results for error in results[1]]

        # Step 3: Save Results
        cluster_utils.ensure_storage_connection()
        self.problems_df = pd.DataFrame(self.problems)
        self.problems_df.to_csv(f"{self.results_folder}problems.csv")
        with open(self.results_folder + "errors.txt", "wt") as f:
            f.write("\n".join(self.errors))

        # Step 4: Analyse Generation
        GenerationRunner.analyse_performance(self.results_folder)
        GenerationRunner.plot_starts_and_targets(self.results_folder)
        GenerationRunner.plot_target_dates_histogram(self.results_folder)

    # Memory Leak of HJ Planner: Only run on workers and with a new process each
    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-remote
    @staticmethod
    @ray.remote(max_calls=1)
    def generate_batch_ray(
        results_folder: str,
        mission_generation_config: dict,
        group: int,
        batch: int,
        verbose,
    ):
        # Step 1: Supress Loggers & Warnings
        if verbose > 3:
            set_arena_loggers(logging.INFO)
            silence_ray_and_tf()
        else:
            set_arena_loggers(logging.WARNING)

        from ocean_navigation_simulator.reinforcement_learning.missions.MissionGenerator import (
            MissionGenerator,
        )

        # Step 2: Prepare Batch Folder
        cluster_utils.ensure_storage_connection()
        batch_folder = f"{results_folder}groups/group_{group}/batch_{batch}/"
        os.makedirs(batch_folder, exist_ok=True)

        try:
            if verbose > 0:
                print(f"GenerationRunner: Starting Batch {batch} Group {group}")

            # Step 3: Generate Batch
            problem_factory = MissionGenerator(
                config=mission_generation_config
                | {
                    "seed": batch,
                    "cache_folder": batch_folder,
                },
                verbose=verbose - 1,
            )
            problems, performance, errors = problem_factory.cache_batch()

            # Step 4: Format Batch Results
            batch_info = (
                {
                    "group": group,
                    "batch": batch,
                }
                | performance
                | get_process_information_dict()
            )
            batch_results = [problem.to_dict() | batch_info for problem in problems]

            # Step 5: Save Batch Results
            pd.DataFrame(batch_results).to_csv(f"{batch_folder}problems.csv")

            if len(errors) > 0:
                with open(batch_folder + "errors.txt", "wt") as f:
                    f.write("\n\n".join(errors))

            if verbose > 0:
                print(
                    "GenerationRunner: Finished Batch {b} Group {g} (total: {t:.0f}s, generate: {ge:.0f}s, plotting: {pt:.0f}s, hindcast {ht:.0f}s, forecast {ft:.0f}s), RAM: {ram}".format(
                        b=batch,
                        g=group,
                        t=batch_info["total_time"],
                        ge=batch_info["generate_time"],
                        pt=batch_info["plot_time"],
                        ht=batch_info["hindcast_time"],
                        ft=batch_info["forecast_time"],
                        ram=batch_info["process_ram"],
                    )
                )

        except Exception as e:
            shutil.rmtree(batch_folder, ignore_errors=True)
            if verbose > 0:
                print(bcolors.red(f"GenerationRunner: Aborted Batch {batch} Group {group}"))
                print(e)
            sys.exit()

        return batch_results, errors

    @staticmethod
    def analyze_batches(
        results_folder: str,
        length: int,
    ):
        groups_folder = results_folder + "groups/"

        existing = [
            batch
            for batch in range(length)
            if os.path.exists(
                groups_folder + f"group_{batch // 100}/" + f"batch_{batch}/" + "problems.csv"
            )
        ]
        missing = [
            batch
            for batch in range(length)
            if not os.path.exists(
                groups_folder + f"group_{batch // 100}/" + f"batch_{batch}/" + "problems.csv"
            )
        ]

        print("Existing: ", len(existing))

        print("Missing CSV: ", len(missing))
        print("Missing CSV: ", missing)

        return existing, missing

    @staticmethod
    def rerun_missing_batches(
        config,
        results_folder,
        length,
        verbose,
    ):
        _, missing = GenerationRunner.analyze_batches(results_folder, length)

        ray.get(
            [
                GenerationRunner.generate_batch_ray.options(
                    num_cpus=config["ray_options"]["resources"].get("CPU", 1.0),
                    num_gpus=config["ray_options"]["resources"].get("GPU", 0.0),
                    max_retries=config["ray_options"].get("max_retries", 10),
                    resources={
                        i: config["ray_options"]["resources"][i]
                        for i in config["ray_options"]["resources"]
                        if i != "CPU" and i != "GPU"
                    },
                ).remote(
                    results_folder=results_folder,
                    mission_generation_config=config["mission_generation"],
                    group=int(batch // config["size"]["batches_per_group"]),
                    batch=batch,
                    verbose=verbose,
                )
                for batch in missing
            ]
        )

    @staticmethod
    def combine_batches(results_folder, length):
        batches, _ = GenerationRunner.analyze_batches(results_folder, length)

        # Step 1: Problems
        cluster_utils.ensure_storage_connection()
        batch_dfs = [
            pd.read_csv(results_folder + f"groups/group_{batch//100}/batch_{batch}/problems.csv")
            for batch in batches
        ]
        problems_df = pd.concat(batch_dfs, ignore_index=True)
        problems_df.to_csv(results_folder + "problems.csv")

        # # Step 2: Errors
        # with open(generation + "errors.txt", "wt") as outfile:
        #     for batch in batches:
        #         error_file = generation + f"groups/group_{batch//100}/batch_{batch}/errors.txt"
        #         if os.path.exists(error_file):
        #             with open(error_file, "rt") as infile:
        #                 outfile.write("\n" + infile.read())

    @staticmethod
    def analyse_performance(
        results_folder: str,
    ):
        cluster_utils.ensure_storage_connection()
        problems_df = pd.read_csv(f"{results_folder}problems.csv")

        ram_numerical = np.array(
            [
                float(row["process_ram"].removesuffix("MB").replace(",", ""))
                for index, row in problems_df.iterrows()
            ]
        )
        print(f"RAM Min:  {ram_numerical.min():.1f}")
        print(f"RAM Mean: {ram_numerical.mean():.1f}")
        print(f"RAM Max:  {ram_numerical.max():.1f}")

        time_numerical = np.array([row["total_time"] for index, row in problems_df.iterrows()])
        print(f"Batch Time Min:  {time_numerical.min():.1f}s")
        print(f"Batch Time Mean: {time_numerical.mean():.1f}s")
        print(f"Batch Time Max:  {time_numerical.max():.1f}s")

    @staticmethod
    def plot_starts_and_targets(
        results_folder: str, scenario_file: str = None, scenario_config: dir = None, c3=None
    ):
        # Step 1: Load Problems and Config
        if results_folder.startswith("/seaweed-storage/"):
            cluster_utils.ensure_storage_connection()
        problems_df = pd.read_csv(f"{results_folder}problems.csv")
        target_df = problems_df[problems_df["factory_index"] == 0]
        analysis_folder = f"{results_folder}analysis/"
        os.makedirs(analysis_folder, exist_ok=True)

        problem = CachedNavigationProblem.from_pandas_row(problems_df.iloc[0])

        # Step 2:
        arena = ArenaFactory.create(
            # only use hindcast
            scenario_file=scenario_file,
            scenario_config=scenario_config,
            problem=problem,
            throw_exceptions=False,
            c3=c3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
                # Plot currents over full GOM
                # HYCOM HC: lon [-98.0,-76.4000244140625], lat[18.1200008392334,31.92000007629394]
                x_interval=[-97.9, -76.41],
                y_interval=[18.13, 31.92],
                time=problem.start_state.date_time,
                spatial_resolution=0.2,
                return_ax=True,
                figsize=(32, 24),
            )
        ax.scatter(
            problems_df["x_0_lon"], problems_df["x_0_lat"], c="red", marker="o", s=6, label="starts"
        )
        ax.scatter(
            target_df["x_T_lon"], target_df["x_T_lat"], c="green", marker="x", s=12, label="targets"
        )
        ax.legend()
        ax.get_figure().savefig(f"{analysis_folder}starts_and_targets.png")
        ax.get_figure().show()

    # @staticmethod
    # def animate_starts_and_targets(generation):
    #     problem_s = config['mission_generation']['problem_timeout'].total_seconds()
    #
    #     def add_ax_func(ax, posix_time):
    #         cur_df = target_df[pd.Timestamp(posix_time) < target_df["t_0"] & target_df["t_0"] < pd.Timestamp(posix_time + problem_s)]
    #
    #         ax.scatter(
    #             problems_df["x_0_lon"], problems_df["x_0_lat"], c="red", marker="o", s=6, label="starts"
    #         )
    #         ax.scatter(
    #             target_df["x_T_lon"], target_df["x_T_lat"], c="green", marker="x", s=12, label="targets"
    #         )
    #
    #     arena.ocean_field.hindcast_data_source.animate_data(
    #         x_interval=[-98.0, -76.4000244140625],
    #         y_interval=[18.1200008392334, 31.92000007629394],
    #         t_interval=[
    #             config['mission_generation']['t_range'][0],
    #             config['mission_generation']['t_range'][1]
    #
    #         ],
    #         temporal_resolution=3600 * 24,
    #         add_ax_func=add_ax_func,
    #         output=f"{analysis_folder}starts_and_targets.gif",
    #     )

    @staticmethod
    def plot_target_dates_histogram(results_folder=None, problems_df=None):
        # Step 1: Load Data
        if results_folder:
            if results_folder.startswith("/seaweed-storage/"):
                cluster_utils.ensure_storage_connection()
            problems_df = pd.read_csv(f"{results_folder}problems.csv")
            analysis_folder = f"{results_folder}analysis/"
            os.makedirs(analysis_folder, exist_ok=True)

        # Step 2: Prepare Data
        if "factory_index" in list(problems_df.columns.values):
            target_df = problems_df[problems_df["factory_index"] == 0]
        else:
            target_df = problems_df
        target_df.insert(2, "t_0_pd", pd.to_datetime(target_df.loc[:, "t_0"]))
        target_df.insert(3, "t_0_ymd", target_df["t_0_pd"].map(lambda x: x.strftime("%Y-%m-%d")))
        df = target_df

        # Step 3: Plot
        fig, ax = plt.subplots(figsize=(16, 12))

        days = pd.date_range(
            df["t_0_pd"].min() - pd.DateOffset(days=1),
            df["t_0_pd"].max() + pd.DateOffset(day=1),
            freq="D",
        )
        months = pd.date_range(
            df["t_0_pd"].min() - pd.DateOffset(months=1),
            df["t_0_pd"].max() + pd.DateOffset(months=1),
            freq="MS",
        )

        df["t_0_pd"].astype(np.int64).plot.hist(ax=ax, bins=days.astype(np.int64))

        ax.set_xticks(months.astype(np.int64).to_list())
        ax.set_xticklabels(months.strftime("%Y-%m").to_list(), rotation=45)

        unique_days = len(df["t_0_ymd"].unique())
        fig.suptitle(f"Target Day Histogram (Days: {unique_days})")

        # fig.savefig(f"{analysis_folder}target_day_histogram.png")
        fig.show()

    @staticmethod
    def plot_ttr_histogram(results_folder):
        # Step 1: Load Data
        if results_folder.startswith("/seaweed-storage/"):
            cluster_utils.ensure_storage_connection()
        df = pd.read_csv(f"{results_folder}problems.csv")
        analysis_folder = results_folder + "analysis/"
        os.makedirs(analysis_folder, exist_ok=True)

        if "random" in df:
            df = df[~df["random"]]

        if "ttr_in_h" in df:
            ttr = df["ttr_in_h"].tolist()
        elif "optimal_time_in_h" in df:
            ttr = df["optimal_time_in_h"].tolist()
        else:
            return

        # Step 2: Plot
        plt.figure()
        plt.hist(ttr, bins=100)
        plt.title("Mission Time-To-Reach Histogram")
        plt.savefig(analysis_folder + "ttr.png", dpi=300)
        plt.show()
