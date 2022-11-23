import datetime
import logging
import os
import time
from typing import Optional

import pandas as pd
import pytz
import ray
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.reinforcement_learning.missions.CachedNavigationProblem import (
    CachedNavigationProblem,
)
from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)
from ocean_navigation_simulator.utils import cluster_utils
from ocean_navigation_simulator.utils.misc import (
    bcolors,
    get_process_information_dict,
    set_arena_loggers,
    silence_ray_and_tf,
    timing,
    timing_dict,
)

sns.set_theme()


class EvaluationRunner:
    """The EvaluationRunner runs distributed evaluation runs of various controllers"""

    def __init__(
        self,
        name,
        config: dict,
        verbose: Optional[int] = 0,
    ):
        self.config = config
        self.verbose = verbose

        #  Step 1: Initialize
        time_string = datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime(
            "%Y_%m_%d_%H_%M_%S"
        )
        cluster_utils.ensure_storage_connection()
        problems = FileProblemFactory(
            csv_file=f"{config['missions']['folder']}problems.csv",
            filter=config["missions"].get("filter", {}),
            seed=config["missions"].get("seed", None),
        ).get_problem_list()
        if verbose > 0:
            print(f"EvaluationRunner: Running {len(problems)} Missions")

        # Step 2: Run
        ray_results = ray.get(
            [
                self.evaluation_run.options(
                    num_cpus=config["ray_options"]["resources"]["CPU"],
                    num_gpus=config["ray_options"]["resources"]["GPU"],
                    max_retries=config["ray_options"]["max_retries"],
                    resources={
                        i: config["ray_options"]["resources"][i]
                        for i in config["ray_options"]["resources"]
                        if i != "CPU" and i != "GPU"
                    },
                ).remote(
                    i,
                    config=config,
                    problem=problem,
                    verbose=verbose - 1,
                )
                for i, problem in enumerate(problems)
            ]
        )
        results_df = pd.DataFrame([r for r in ray_results if r is not False])
        if verbose > 0:
            print(
                f"EvaluationRunner: Finished {results_df.shape[0]}/{len(problems)} Missions sucessfully."
            )

        results_folder = config["controller"].get(
            "experiment",
            "/seaweed-storage/evaluation/" + config["controller"]["name"] + "/",
        )
        os.makedirs(results_folder, exist_ok=True)
        csv_file = results_folder + f"eval_{name}_{len(problems)}m_{time_string}.csv"
        results_df.to_csv(csv_file)

        print(results_folder)
        self.print_results(csv_file)

        # Step 3: Weights & Biases
        # if config["wandb"]["fake_iterations"]:
        #     self.fake_wandb_iterations()

        if config["wandb"]["upload_summary"]:
            if not config["wandb"].get("run_id", False):
                with open(f"{config['controller']['experiment']}wandb_run_id", "rt") as f:
                    wandb_run_id = f.read()
            else:
                config["wandb"]["run_id"]
            self.update_wandb_summary(
                csv_file=csv_file,
                wandb_run_id=wandb_run_id,
                time_string=time_string,
            )

    # @staticmethod
    # def fake_wandb_iterations(csv_file, config, iterations=300):
    #     df = pd.read_csv(csv_file, index_col=0)
    #     wandb.init(
    #         project="seaweed-rl",
    #         entity="jeromejeannin",
    #         dir="/seaweed-storage/",
    #         config=config,
    #     )
    #     wandb.run.summary["validation_1000"] = {
    #         "date": time_string,
    #         "success": df["success"].mean(),
    #         "running_time": df["running_time"].mean(),
    #     }
    #     wandb.finish()

    @staticmethod
    def update_wandb_summary(csv_file, wandb_run_id, time_string, indexes=None):
        df = pd.read_csv(csv_file, index_col=0)
        if indexes is not None:
            df = df[df["index"].isin(indexes)]
        wandb.init(
            project="seaweed-rl",
            entity="jeromejeannin",
            dir="/tmp",
            id=wandb_run_id,
            resume="must",
        )
        wandb.run.summary[f"validation_{df.shape[0]}"] = {
            "date": time_string,
            "length": df.shape[0],
            "success": df["success"].mean(),
            "running_time": df["running_time"].mean(),
        }
        wandb.finish()

    @staticmethod
    def print_results(csv_file, indexes=None):
        df = pd.read_csv(csv_file, index_col=0)

        if indexes is not None:
            df = df[df["index"].isin(indexes)]

        if indexes is None or df.shape[0] == len(indexes):
            print(f"Indexes: {df.shape[0]} / {len(indexes) if indexes is not None else 0}")
            print(f"Success Mean: {df['success'].mean():.2%}")
            print(f"Success Std: {df['success'].std():.2%}")
            print(f"Running_time: {df['running_time'].mean()} h")

            # folder = os.path.dirname(csv_file)
            # with open(folder + "/evaluation.txt", "at") as f:
            #     f.write(
            #         f"File: {csv_file}, Indexes: [Min: {min(indexes)}, Max: {max(indexes)}, Length: {len(indexes)}]\n"
            #     )

    @staticmethod
    @ray.remote(max_calls=1)
    def evaluation_run(
        i,
        config: dict,
        problem: CachedNavigationProblem,
        verbose: int = 0,
    ):

        cluster_utils.ensure_storage_connection()

        if verbose < 3:
            set_arena_loggers(logging.ERROR)
            silence_ray_and_tf()
        else:
            set_arena_loggers(logging.INFO)

        try:
            mission_start_time = time.time()
            if verbose > 0:
                print(
                    "EvaluationRunner: Starting Mission {i} (I{I}, G{gr} B{b} FI{fi})".format(
                        i=i,
                        I=problem.extra_info["index"],
                        gr=problem.extra_info["group"],
                        b=problem.extra_info["batch"],
                        fi=problem.extra_info["factory_index"],
                    )
                )

            # Step 1: Create Arena
            with timing("EvaluationRunner: Created Controller ({})", verbose - 1):
                arena = ArenaFactory.create(
                    scenario_file=config["scenario_file"],
                    problem=problem,
                    x_interval=problem.extra_info["x_cache"],
                    y_interval=problem.extra_info["y_cache"],
                )
                observation = arena.reset(platform_state=problem.start_state)
                problem_status = arena.problem_status(problem=problem)

            # Step 2: Create Controller
            with timing("EvaluationRunner: Created Controller ({})", verbose - 1):
                problem.platform_dict = arena.platform.platform_dict

                if config["controller"]["name"] == "CachedHJReach2DPlannerForecast":
                    controller = problem.get_cached_forecast_planner(
                        base_path=config["missions"]["folder"], arena=arena
                    )
                elif config["controller"]["name"] == "CachedHJReach2DPlannerHindcast":
                    controller = problem.get_cached_hindcast_planner(
                        base_path=config["missions"]["folder"], arena=arena
                    )
                elif config["controller"]["type"].__name__ == "RLController":
                    controller = config["controller"]["type"](
                        config=config, problem=problem, arena=arena
                    )
                else:
                    controller = config["controller"]["type"](
                        problem=problem, **config["controller"].get("kwargs", {})
                    )

            # Step 3: Run Arena
            performance = {"arena": 0, "controller": 0}
            with timing("EvaluationRunner: Run Arena ({})", verbose - 1):
                steps = 0
                while problem_status == 0:
                    with timing_dict(performance, "controller"):
                        if config["controller"]["name"] == "CachedHJReach2DPlannerHindcast":
                            action = controller.get_action(
                                observation.replace_datasource(
                                    arena.ocean_field.hindcast_data_source
                                )
                            )
                        else:
                            action = controller.get_action(observation)
                    with timing_dict(performance, "arena"):
                        observation = arena.step(action)
                        problem_status = arena.problem_status(problem=problem)
                        steps += 1

            # Step 4: Format Results
            result = (
                {
                    "index": problem.extra_info["index"],
                    "group": problem.extra_info["group"],
                    "batch": problem.extra_info["batch"],
                    "factory_index": problem.extra_info["factory_index"],
                    "controller": type(controller).__name__,
                    "success": True if problem_status > 0 else False,
                    "problem_status": problem_status,
                    "steps": steps,
                    "running_time": problem.passed_seconds(observation.platform_state),
                    "minimal_distance": problem.distance(observation.platform_state).deg,
                    "final_distance": problem.distance(observation.platform_state).deg,
                }
                | {
                    "process_time": f"{time.time() - mission_start_time:.2f}s",
                    "arena_time": f"{performance['arena']:.2f}s",
                    "controller_time": f"{performance['controller']:.2f}s",
                }
                | get_process_information_dict()
            )

            if verbose > 0:
                text = arena.problem_status_text(problem_status)
                status = bcolors.green(text) if problem_status > 0 else bcolors.red(text)
                print(
                    "EvaluationRunner: Finished Mission {i} (I{I}, G{gr} B{b} FI{fi})".format(
                        i=i,
                        I=problem.extra_info["index"],
                        gr=problem.extra_info["group"],
                        b=problem.extra_info["batch"],
                        fi=problem.extra_info["factory_index"],
                    ),
                    "({status}, {steps} Steps, {running_time:.1f}h, ttr: {ttr:.1f}h, {dist:.2f}°)".format(
                        status=status,
                        steps=steps,
                        running_time=result["running_time"] / 3600,
                        ttr=problem.extra_info["ttr_in_h"],
                        dist=result["final_distance"],
                    ),
                    "(Total: {total}, Arena: {ar}, Controller: {co}, RAM: {ram}, GPU: {gpu})".format(
                        total=result["process_time"],
                        ar=result["arena_time"],
                        co=result["controller_time"],
                        ram=result["process_ram"],
                        gpu=result["process_gpu_used"],
                    ),
                )

            return result

        except Exception as e:
            if verbose > 0:
                print(
                    "EvaluationRunner: Aborted Mission {i} (I{I}, G{gr} B{b} FI{fi})".format(
                        i=i,
                        I=problem.extra_info["index"],
                        gr=problem.extra_info["group"],
                        b=problem.extra_info["batch"],
                        fi=problem.extra_info["factory_index"],
                    )
                )
                raise e

            return False

    @staticmethod
    def plot_confusion(csv_file1, csv_file2, label1="left", label2="right"):
        df_1 = pd.read_csv(csv_file1, index_col=0)
        df_2 = pd.read_csv(csv_file2, index_col=0)

        df_mixed = df_1.merge(
            df_2, how="inner", left_on="index", right_on="index", suffixes=("_left", "_right")
        )

        confusion_matrix = pd.crosstab(
            df_mixed["success_left"],
            df_mixed["success_right"],
            rownames=[label1],
            colnames=[label2],
        )

        ax = sns.heatmap(
            confusion_matrix, annot=True, cmap=plt.cm.RdYlGn, square=True, norm=LogNorm(), fmt="g"
        )
        plt.tick_params(
            axis="both", which="major", labelbottom=False, bottom=False, top=True, labeltop=True
        )
        ax.xaxis.set_label_position("top")
        plt.show()

    @staticmethod
    def plot_mission_time_and_success(df, name = '', file = None):
        if "random" in df:
            df = df[~df["random"]]

        if "ttr_in_h" in df:
            ttr_succ = df[df["success"]]["ttr_in_h"].tolist()
            ttr_fail = df[~df["success"]]["ttr_in_h"].tolist()
        elif "optimal_time_in_h" in df:
            ttr_succ = df[df["success"]]["optimal_time_in_h"].tolist()
            ttr_fail = df[~df["success"]]["optimal_time_in_h"].tolist()
        else:
            return

        # Step 2: Plot
        plt.figure()
        plt.hist(ttr_succ, bins=100, alpha=0.5, color="g", label="success")
        plt.hist(ttr_fail, bins=100, alpha=0.5, color="r", label="failed")
        plt.title(f"Mission Time-To-Reach Histogram ({name})")
        if file is not None:
            plt.savefig(file, dpi=300)
        plt.show()
