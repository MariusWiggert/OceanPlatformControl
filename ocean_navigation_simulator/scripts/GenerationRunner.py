import datetime
import json
import os
import pickle
import shutil
import socket
import sys
import time
from typing import Optional
import pandas as pd
import psutil
import pytz
import ray
import requests
import pynvml


from ocean_navigation_simulator.problem_factories.ShortMissionProblemFactory import ShortMissionProblemFactory
from ocean_navigation_simulator.scripts.RayUtils import RayUtils


class GenerationRunner:
    def __init__(
        self,
        name: str,
        scenario_name: str,
        runs: int,
        num_batches_per_run: int,
        batch_size: int,
        problem_factory_config: dict,
        verbose: Optional[int] = 0
    ):
        self.name = name
        self.scenario_name = scenario_name
        self.runs = runs
        self.num_batches_per_run = num_batches_per_run
        self.batch_size = batch_size
        self.problem_factory_config = problem_factory_config
        self.verbose = verbose

        # Step 1: Prepare Paths
        self.timestring = datetime.datetime.now(datetime.now(tz=pytz.timezone('US/Pacific'))).strftime("%Y_%m_%d_%H_%M_%S")
        self.results_folder = f'/seaweed-storage/generation/{self.name}_{self.timestring}/'

        # Step 2: Save configuration
        RayUtils.check_storage_connection()
        os.makedirs(self.results_folder)
        pickle.dump(self.problem_factory_config, open(f'{self.results_folder}config.pickle', 'wb'))

        # Step 3: Run Generation with Ray
        self.ray_results = ray.get([self.generation_run.remote(
            self.results_folder, scenario_name, problem_factory_config, run, num_batches_per_run, batch_size, verbose
        ) for run in range(runs)])
        self.problems = [problem for problems in self.ray_results for problem in problems]

        # Step 4: Save Results
        RayUtils.check_storage_connection()
        self.results_df = pd.DataFrame(self.problems)
        self.results_df.to_csv(f'{self.results_folder}problems.csv')

    @staticmethod
    @ray.remote(num_cpus=1, num_gpus=1, max_retries=10)
    def generation_run(results_folder, scenario_name, problem_factory_config, seed, num_batches_per_run, batch_size, verbose):
        try:
            run_folder = f'{results_folder}/seed_{seed}'
            run_start_time = time.time()
            run_results = []
            seed_info = {
                'seed_pid': os.getpid(),
                'seed_public_ip': requests.get('https://api.ipify.org').content.decode('utf8'),
                'seed_private_ip': socket.gethostbyname(socket.gethostname()),
            }
            problem_factory = ShortMissionProblemFactory(
                scenario_name=scenario_name,
                config={'seed': seed} | problem_factory_config,
                verbose=verbose-1,
            )

            for batch in range(num_batches_per_run):
                batch_start_time = time.time()
                batch_folder = f'{run_folder}/batch_{batch}/'

                # Step 1: Generate Batch
                RayUtils.check_storage_connection()
                os.makedirs(batch_folder, exist_ok = True)
                problems = problem_factory.generate_batch(batch_size)
                problem_factory.plot_batch(batch_size, filename=f'{batch_folder}animation.gif')
                problem_factory.hindcast_planner.save_plan(batch_folder)

                # Step 2: Format Results
                batch_time = time.time() - batch_start_time
                pynvml.nvmlInit()
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
                batch_results = [problem.to_dict() | seed_info | {
                    'batch': batch,
                    'batch_time': f'{batch_time:.2f}s',
                    'batch_ram': f'{psutil.Process().memory_info().rss / 1e6:,.0f}MB',
                    'batch_gpu_total': f'{gpu_info.total / 1e6:,.0f}MB',
                    'batch_gpu_used': f'{gpu_info.free / 1e6:,.0f}MB',
                    'batch_gpu_free': f'{gpu_info.used / 1e6:,.0f}MB',
                } for index, problem in enumerate(problems)]

                # Step 3: Save Results
                RayUtils.check_storage_connection()
                os.makedirs(batch_folder, exist_ok = True)
                pd.DataFrame(batch_results).to_csv(f'{batch_folder}/problems.csv')
                run_results.extend(batch_results)

                if verbose > 0:
                    print(f'GenerationRunner[{seed}]: Finished Batch {batch} ({time.time() - batch_start_time:.1f}s)')

            RayUtils.check_storage_connection()
            os.makedirs(run_folder, exist_ok = True)
            pd.DataFrame(run_results).to_csv(f'{run_folder}/problems.csv')

            if verbose > 0:
                print(f'GenerationRunner[{seed}]: Finished Run ({time.time()-run_start_time:.1f}s)')

        except Exception as e:
            shutil.rmtree(run_folder, ignore_errors=True)
            print(e)
            sys.exit()

        return run_results