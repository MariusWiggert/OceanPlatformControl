import datetime
import logging
import os
import sys
import time
from typing import Optional

import pandas as pd
import pytz
import ray

# import wandb
import wandb

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
    timing,
    set_arena_loggers,
    silence_ray_and_tf,
)


class EvaluationRunner:
    """The EvaluationRunner runs distributed evaluation runs of various controllers"""

    def __init__(
        self,
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
        results_df = pd.DataFrame(ray_results)

        results_folder = config["controller"].get(
            "experiment",
            "/seaweed-storage/evaluation/" + config["controller"]["name"] + "_" + time_string + "/",
        )
        os.makedirs(results_folder + "evaluation/", exist_ok=True)
        csv_file = results_folder + f"evaluation/evaluation_{time_string}.csv"
        results_df.to_csv(csv_file)

        self.print_results(csv_file)

        # Step 3: Weights & Biases
        if config['wandb']['fake_iterations']:
            self.fake_wandb_iterations()

        # if config['wandb']['upload_summary']:
        #     with open(f"{config['experiment']}wandb_run_id", "rt") as f:
        #         wandb_run_id = f.read()
        #     self.update_wandb_summary(
        #         csv_file=f"{config['experiment']}evaluation/evaluation_{time_string}.csv",
        #         wandb_run_id=wandb_run_id,
        #         time_string=time_string,
        #     )

    @staticmethod
    def fake_wandb_iterations(csv_file, config, iterations=300):
        df = pd.read_csv(csv_file, index_col=0)
        wandb.init(
            project="seaweed-rl",
            entity="jeromejeannin",
            dir="/seaweed-storage/",
            config=config,
        )
        wandb.run.summary["validation_1000"] = {
            "date": time_string,
            "success": df["success"].mean(),
            "running_time": df["running_time"].mean(),
        }
        wandb.finish()

    @staticmethod
    def update_wandb_summary(csv_file, wandb_run_id, time_string):
        df = pd.read_csv(csv_file, index_col=0)
        wandb.init(
            project="RL for underactuated navigation",
            entity="ocean-platform-control",
            dir="/seaweed-storage/",
            id=wandb_run_id,
            resume="must",
        )
        wandb.run.summary["validation_1000"] = {
            "date": time_string,
            "success": df["success"].mean(),
            "running_time": df["running_time"].mean(),
        }
        wandb.finish()

    @staticmethod
    def print_results(csv_file):
        df = pd.read_csv(csv_file, index_col=0)

        print("success:", df["success"].mean())
        print("running_time:", df["running_time"].mean())


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
            set_arena_loggers(logging.WARNING)
            silence_ray_and_tf()
        else:
            set_arena_loggers(logging.INFO)

        try:
            mission_start_time = time.time()
            if verbose > 0:
                print(f'EvaluationRunner: Started Mission {problem.extra_info["index"]:03d}')

            # Step 1: Create Arena
            with timing("EvaluationRunner: Created Controller ({{:.1f}}s)", verbose - 1):
                arena = ArenaFactory.create(
                    scenario_file=config["scenario_file"], problem=problem, verbose=verbose - 2
                )
                observation = arena.reset(platform_state=problem.start_state)
                problem_status = arena.problem_status(problem=problem)

            # Step 2: Create Controller
            with timing("EvaluationRunner: Created Controller ({{:.1f}}s)", verbose - 1):
                problem.platform_dict = arena.platform.platform_dict

                if config["controller"]["name"] == "CachedHJReach2DPlannerForecast":
                    controller = problem.get_cached_forecast_planner(config["missions"]["folder"])
                elif config["controller"]["name"] == "CachedHJReach2DPlannerHindcast":
                    controller = problem.get_cached_hindcast_planner(config["missions"]["folder"])
                elif config["controller"]["type"].__name__ == "RLController":
                    controller = config["controller"](
                        config=config,
                        problem=problem,
                        arena=arena,
                        verbose=verbose - 2,
                    )
                else:
                    controller = config["controller"]["type"](problem=problem, verbose=verbose - 2)

            # Step 3: Run Arena
            with timing("EvaluationRunner: Run Arena ({{:.1f}}s)", verbose - 1):
                steps = 0
                while problem_status == 0:
                    if config["controller"]["name"] == "CachedHJReach2DPlannerHindcast":
                        action = controller.get_action(
                            observation.replace_datasource(arena.ocean_field.hindcast_data_source)
                        )
                    else:
                        action = controller.get_action(observation)
                    observation = arena.step(action)
                    problem_status = arena.problem_status(problem=problem)
                    steps += 1

            # Step 4: Format Results
            result = (
                {
                    "index": problem.extra_info["index"],
                    "controller": type(controller).__name__,
                    "success": True if problem_status > 0 else False,
                    "problem_status": problem_status,
                    "steps": steps,
                    "running_time": problem.passed_seconds(observation.platform_state),
                    "final_distance": problem.distance(observation.platform_state).deg,
                }
                | {
                    "process_time": f"{time.time() - mission_start_time:.2f}s",
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
                    "(Process: {time}, RAM: {ram}, GPU: {gpu})".format(
                        time=result["process_time"],
                        ram=result["process_ram"],
                        gpu=result["process_gpu_used"],
                    ),
                )

            return result

        except Exception as e:
            if verbose > 0:
                print(f'EvaluationRunner: Aborted Mission {problem.extra_info["index"]:03d} ######')
                print(e)
            sys.exit()
