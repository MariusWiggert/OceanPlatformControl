import datetime
import logging
import os
import sys
import time
from typing import Optional, Callable
import pandas as pd
import psutil
import pytz
import ray
import requests
import shutil
import torch
import wandb

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils
from ocean_navigation_simulator.utils.bcolors import bcolors


class EvaluationRunner:
    def __init__(
            self,
            config: dict,
            verbose: Optional[int] = 0,
    ):
        self.config = config
        self.verbose = verbose

        time_string = datetime.datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y_%m_%d_%H_%M_%S")
        problem_factory = FileMissionProblemFactory(csv_file=f"{config['missions']['folder']}problems.csv", limit=config['missions']['limit'])
        problems = problem_factory.get_problem_list()

        if verbose > 0:
            print(f'EvaluationRunner: Running {len(problems)} Missions')

        ray_results = ray.get([self.evaluation_run.options(
            num_cpus=config["ray_options"]['resources']['CPU'],
            num_gpus=config["ray_options"]['resources']['GPU'],
            max_retries=config["ray_options"]['max_retries'],
            resources={i:config["ray_options"]['resources'][i] for i in config["ray_options"]['resources'] if i!='CPU' and i!= "GPU"}
        ).remote(
            config=config,
            problem=problem,
            verbose=verbose-1,
        ) for problem in problems])

        results_df = pd.DataFrame(ray_results).set_index('index').rename_axis(None)
        os.makedirs(f"{config['experiment']}evaluation/", exist_ok=True)
        results_df.to_csv(f"{config['experiment']}evaluation/evaluation_{time_string}.csv")

        with open(f"{config['experiment']}wandb_run_id", 'rt') as f:
            wandb_run_id = f.read()

        self.update_wandb_summary(
            csv_file=f"{config['experiment']}evaluation/evaluation_{time_string}.csv",
            wandb_run_id=wandb_run_id,
        )

    @staticmethod
    def update_wandb_summary(csv_file, wandb_run_id):
        df = pd.read_csv(csv_file, index_col=0)

        wandb.init(
            project="RL for underactuated navigation",
            entity="ocean-platform-control",
            id=wandb_run_id,
            resume="must"
        )

        wandb.run.summary["validation_1000"] = {
            'success': df['success'].mean(),
            'running_time': df['running_time'].mean()
        }

        wandb.finish()

    @staticmethod
    @ray.remote(max_calls=1)
    def evaluation_run(
        config: dict,
        problem: NavigationProblem,
        verbose:  int = 0,
    ):
        try:
            mission_start_time = time.time()
            if verbose > 0:
                print(f'EvaluationRunner: Started Mission {problem.extra_info["index"]:03d}')

            # Step 1: Create Arena
            with Utils.timing(f'EvaluationRunner: Created Controller ({{:.1f}}s)', verbose-1):
                arena = ArenaFactory.create(scenario_name=config['scenario_name'], problem=problem, verbose=verbose-2)
                observation = arena.reset(platform_state=problem.start_state)
                problem_status = arena.problem_status(problem=problem)

            # Step 2: Create Controller
            with Utils.timing(f'EvaluationRunner: Created Controller ({{:.1f}}s)', verbose-1):
                problem.platform_dict = arena.platform.platform_dict
                if config['controller'].__name__ == 'RLController':
                    controller = config['controller'](
                        config=config,
                        problem=problem,
                        arena=arena,
                        verbose=verbose-2,
                    )
                else:
                    controller = config['controller'](problem=problem, verbose=verbose-2)

            # Step 3: Run Arena
            with Utils.timing(f'EvaluationRunner: Run Arena ({{:.1f}}s)', verbose-1):
                steps = 0
                while problem_status==0:
                    action = controller.get_action(observation)
                    observation = arena.step(action)
                    problem_status = arena.problem_status(problem=problem)
                    steps += 1

            result = {
                'index': problem.extra_info['index'],
                'controller': type(controller).__name__,
                'success': True if problem_status==1 else False,
                'problem_status': problem_status,
                'steps': steps,
                'running_time': problem.passed_seconds(observation.platform_state),
                'final_distance': problem.distance(observation.platform_state),
            } | {
                'process_time': f'{time.time() - mission_start_time:.2f}s',
            } | Utils.get_process_information_dict()

            if verbose > 0:
                status = f"{bcolors.OKGREEN}Success{bcolors.ENDC}" if problem_status > 0 else f"{bcolors.FAIL}Timeout{bcolors.ENDC}" if problem_status == -1 else (f"{bcolors.FAIL}Stranded{bcolors.ENDC}" if problem_status == -2 else f"{bcolors.FAIL}Outside Arena{bcolors.ENDC}")
                print(
                    f'EvaluationRunner: Finished Mission {result["index"]:03d} with {status}',
                    f'({steps} Steps, {result["running_time"]/(3600*24):.0f}d {result["running_time"] % (3600*24) / 3600:.0f}h, {result["final_distance"]:.4f} Degree)',
                    f'(Process: {result["process_time"]}, RAM: {result["process_ram"]}, GPU: {result["process_gpu_used"]})',
                )

            return result

        except Exception as e:
            if verbose > 0:
                print(f'EvaluationRunner: Aborted Mission {problem.extra_info["index"]:03d} ######')
                print(e)
            sys.exit()