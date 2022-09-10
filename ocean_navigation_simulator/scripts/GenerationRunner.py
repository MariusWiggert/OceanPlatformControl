import os
import datetime
import pickle
import shutil
import socket
import sys
import time
from types import SimpleNamespace
from typing import Optional
import numpy as np
import pandas as pd
import psutil
import pytz
import ray
import pynvml
import os
import logging

from matplotlib import pyplot as plt

from ocean_navigation_simulator.problem_factories.ShortMissionProblemFactory import ShortMissionProblemFactory
from ocean_navigation_simulator.scripts.Utils import Utils

class GenerationRunner:
    def __init__(
        self,
        name: str,
        scenario_name: str,
        groups: int,
        batches_per_group: int,
        batch_size: int,
        problem_factory_config: dict,
        ray_options: dict,
        verbose: Optional[int] = 0
    ):
        self.name = name
        self.scenario_name = scenario_name
        self.groups = groups
        self.batches_per_group = batches_per_group
        self.batch_size = batch_size
        self.problem_factory_config = problem_factory_config
        self.verbose = verbose

        # Step 1: Prepare Paths
        self.timestring = datetime.datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y_%m_%d_%H_%M_%S")
        self.results_folder = f'/seaweed-storage/generation/{self.name}_{self.timestring}/'

        # Step 2: Save configuration
        Utils.ensure_storage_connection()
        os.makedirs(self.results_folder)
        with open(f'{self.results_folder}config.pickle', 'wb') as f:
            pickle.dump(self.problem_factory_config, f)

        # Step 3: Run Generation with Ray
        if verbose > 0:
            print(f'GenerationRunner: Generating {groups * batches_per_group} batches in {groups} groups')
        self.ray_results = ray.get([self.generate_batch.options(
            num_cpus=ray_options['resources']['CPU'],
            num_gpus=ray_options['resources']['GPU'],
            max_retries=ray_options['max_retries'],
            resources={i:ray_options['resources'][i] for i in ray_options['resources'] if i!='CPU' and i!= "GPU"}
        ).remote(
            results_folder=self.results_folder,
            scenario_name=scenario_name,
            problem_factory_config=problem_factory_config,
            group=int(batch // batches_per_group),
            batch=batch,
            batch_size=batch_size,
            verbose=verbose,
        ) for batch in range(groups * batches_per_group)])
        self.problems = [problem for problems in self.ray_results for problem in problems]

        # Step 4: Save Results
        Utils.ensure_storage_connection()
        self.problems_df = pd.DataFrame(self.problems)
        self.problems_df.to_csv(f'{self.results_folder}problems.csv')

        # Step 5: Analyse Generation
        GenerationRunner.analyse_generation(self.results_folder)
        GenerationRunner.plot_generation(self.results_folder)


    # Memory Leak of HJ Planner: Only run on workers and with a new process each
    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-remote
    @staticmethod
    @ray.remote(max_calls=1)
    def generate_batch(
        results_folder: str,
        scenario_name: str,
        problem_factory_config: dict,
        group: int,
        batch: int,
        batch_size: int,
        verbose
    ):
        # Supress TF CPU warnings:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        logging.getLogger('tensorflow').setLevel(logging.FATAL)
        logging.getLogger('absl').setLevel(logging.FATAL)
        # Suppress GRPC warnings:
        # https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
        os.environ['GRPC_VERBOSITY'] = 'None' # One of: DEBUG, INFO, ERROR, NONE
        logging.getLogger('chttp2_transport.cc').setLevel(logging.FATAL)

        try:
            batch_start_time = time.time()
            batch_folder = f'{results_folder}/groups/group_{group}/batch_{batch}/'

            if verbose > 0:
                print(f'GenerationRunner: Starting Batch {batch} Group {group}')

            # Step 1: Generate Batch
            generate_start = time.time()
            problem_factory = ShortMissionProblemFactory(
                scenario_name=scenario_name,
                config={'seed': batch} | problem_factory_config,
                verbose=verbose-1,
            )
            problems = problem_factory.generate_batch(batch_size)
            generate_time = time.time()-generate_start

            # Step 2: Plot Batch
            Utils.ensure_storage_connection()
            os.makedirs(batch_folder, exist_ok = True)
            plot_start = time.time()
            problem_factory.plot_batch(batch_size, filename=f'{batch_folder}animation.gif')
            plot_time = time.time()-plot_start

            # step 3: Save Batch Planner
            Utils.ensure_storage_connection()
            problem_factory.hindcast_planner.save_plan(batch_folder)

            # Step 4: Format Batch Results
            try:
                pynvml.nvmlInit()
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
            except Exception as e:
                gpu_info = SimpleNamespace(total=0, free=0, used=0)
            batch_info = {
                'group': group,
                'batch': batch,
                'batch_pid': os.getpid(),
                # 'batch_public_ip': requests.get('https://api.ipify.org').content.decode('utf8'),
                'batch_private_ip': socket.gethostbyname(socket.gethostname()),
                'batch_time': f'{ time.time()-batch_start_time:.2f}s',
                'batch_ram': f'{psutil.Process().memory_info().rss / 1e6:,.0f}MB',
                'batch_gpu_total': f'{gpu_info.total / 1e6:,.0f}MB',
                'batch_gpu_used': f'{gpu_info.free / 1e6:,.0f}MB',
                'batch_gpu_free': f'{gpu_info.used / 1e6:,.0f}MB',
            }
            batch_results = [problem.to_dict() | batch_info for index, problem in enumerate(problems)]

            # Step 5: Save Batch Results
            Utils.ensure_storage_connection()
            pd.DataFrame(batch_results).to_csv(f'{batch_folder}/problems.csv')

            if verbose > 0:
                print(f'GenerationRunner: Finished Batch {batch} Group {group} (total: {time.time()-batch_start_time:.1f}s, generate: {generate_time:.1f}s, plotting: {plot_time:.1f}s)')

        except Exception as e:
            shutil.rmtree(batch_folder, ignore_errors=True)
            if verbose > 0:
                print(f'GenerationRunner: Aborted Batch {batch} Group {group}')
            print(e)
            sys.exit()

        return batch_results

    @staticmethod
    def analyse_generation(
        results_folder: str,
    ):
        Utils.ensure_storage_connection()
        problems_df = pd.read_csv(f'{results_folder}problems.csv')

        ram_numerical = np.array([float(row['batch_ram'].removesuffix('MB').replace(',', '')) for index, row in problems_df.iterrows()])
        print(f'RAM Min:  {ram_numerical.min():.1f}')
        print(f'RAM Mean: {ram_numerical.mean():.1f}')
        print(f'RAM Max:  {ram_numerical.max():.1f}')

        time_numerical = np.array([float(row['batch_time'].removesuffix('s')) for index, row in problems_df.iterrows()])
        print(f'Batch Time Min:  {time_numerical.min():.1f}s')
        print(f'Batch Time Mean: {time_numerical.mean():.1f}s')
        print(f'Batch Time Max:  {time_numerical.max():.1f}s')

    @staticmethod
    def plot_generation(
        results_folder: str,
        n: Optional[int] = 100,
    ):
        # Step 1:
        Utils.ensure_storage_connection()
        problems_df = pd.read_csv(f'{results_folder}problems.csv').head(n=n)
        target_df = problems_df[problems_df['factory_index'] == 0]
        analysis_folder = f'{results_folder}analysis/'
        os.makedirs(analysis_folder, exist_ok=True)

        # Step 2:
        plt.figure(figsize=(12, 12))
        plt.scatter(target_df['x_T_lon'], target_df['x_T_lat'], c='green', marker='x', label='target')
        plt.scatter(problems_df['x_0_lon'], problems_df['x_0_lat'], c='red', marker='o', label='start')
        plt.savefig(f'{analysis_folder}starts_and_targets.png')
        plt.show()