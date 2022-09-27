import datetime
import os
import time
from typing import Optional
import pandas as pd
import psutil
import ray
import requests
import torch

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.ProblemFactory import ProblemFactory


class EvaluationRunner:
    def __init__(
        self,
        scenario_name: str,
        controller_class,
        problem_factory: ProblemFactory,
        verbose: Optional[int] = 0,
    ):
        time_string = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        problems = problem_factory.get_problem_list()

        if verbose > 0:
            print(
                f"EvaluationRunner: running {len(problems)} missions with {controller_class.gpus} gpus"
            )

        ray_results = ray.get(
            [
                self.evaluation_run.options(num_gpus=controller_class.gpus).remote(
                    problem, scenario_name, controller_class, verbose
                )
                for problem in problems
            ]
        )

        results_df = pd.DataFrame(ray_results).set_index("index").rename_axis(None)
        results_folder = (
            f"/seaweed-storage/evaluation/{scenario_name}/{controller_class.__name__}_{time_string}"
        )
        os.makedirs(results_folder, exist_ok=True)
        results_df.to_csv(f"{results_folder}/results.csv")

    @staticmethod
    @ray.remote(num_cpus=1, num_gpus=1)
    def evaluation_run(problem, scenario_name, controller_class, verbose):
        mission_start_time = time.time()
        if verbose > 0:
            print(f'##### Starting Mission {problem.extra_info["index"]:03d} ######')

        # Step 1: Create Arena
        start = time.time()
        arena = ArenaFactory.create(
            scenario_name=scenario_name, problem=problem, verbose=verbose - 1
        )
        observation = arena.reset(platform_state=problem.start_state)
        problem_status = arena.problem_status(problem=problem)
        if verbose > 0:
            print(f"EvaluationRunner: Create Arena ({time.time() - start:.1f}s)")

        # Step 2: Create Controller
        start = time.time()
        controller = controller_class(
            problem=problem, platform_dict=arena.platform.platform_dict, verbose=verbose - 1
        )
        if verbose > 0:
            print(f"EvaluationRunner: Create Controller ({time.time() - start:.1f}s)")

        # Step 3: Run Arena
        start = time.time()
        steps = 0
        while problem_status == 0:
            action = controller.get_action(observation)
            observation = arena.step(action)
            problem_status = arena.problem_status(problem=problem)
            steps += 1

        if verbose > 0:
            print(f"EvaluationRunner: Running Arena ({time.time() - start:.1f}s)")

        result = {
            "index": problem.extra_info["index"],
            "controller": type(controller).__name__,
            "success": True if problem_status == 1 else False,
            "steps": steps,
            "running_time": problem.passed_seconds(observation.platform_state),
            "final_distance": problem.distance(observation.platform_state),
        } | {
            "process_pid": os.getpid(),
            "process_worker": requests.get("https://api.ipify.org").content.decode("utf8"),
            "process_time": f"{time.time() - mission_start_time:.2f}s",
            "process_ram": f"{psutil.Process().memory_info().rss/1e6:.1f}MB",
            "process_cuda_reserved": f"{torch.cuda.memory_reserved(0)/1e6:.1f}MB",
            "process_cuda_allocated": f"{torch.cuda.memory_allocated(0)/1e6:.1f}MB",
            "process_cuda_free": f"{(torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0))/1e6:.1f}MB",
        }

        if verbose > 0:
            print(
                f'##### Finished Mission {result["index"]:03d} with {"Success" if result["success"] else "Failure"}',
                f'({steps} Steps, {result["running_time"]/(3600*24):.0f}d {result["running_time"] % (3600*24) / 3600:.0f}h, {result["final_distance"]:.4f} Degree)'
                f' (Process: {result["process_time"]:.1f}s, RAM: {result["process_ram"] / 1e6:.1f}MB, GPU: {result["process_gpu"] / 1e6:.1f}MB)',
            )

        return result
