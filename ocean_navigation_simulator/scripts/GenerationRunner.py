import json
import os
import pickle
import shutil
import socket
import sys
import time
import pandas as pd
import psutil
import ray
import requests
import pynvml


from ocean_navigation_simulator.problem_factories.ShortMissionProblemFactory import ShortMissionProblemFactory
from ocean_navigation_simulator.scripts.RayUtils import RayUtils


class GenerationRunner:
    def __init__(self, name, scenario_name, runs, num_batches_per_run, batch_size):
        RayUtils.check_storage_connection()

        # Step 1: Create and Clean Results Folder
        results_folder = f'/seaweed-storage/generation_runner/{scenario_name}/{name}/'
        if os.path.exists(results_folder):
            shutil.rmtree(results_folder, ignore_errors=True)
        os.makedirs(results_folder, exist_ok=True)

        # Step 2: Save Mission Config
        with open(f'{results_folder}config.pickle', 'wb') as f:
            pickle.dump(ShortMissionProblemFactory.default_config, f)

        # Step 3: Run Generation with Ray
        ray_results = ray.get([self.generation_run.remote(
            scenario_name, seed, num_batches_per_run, batch_size, results_folder
        ) for seed in range(runs)])
        all_problems = [problem for problems in ray_results for problem in problems]

        # Step 4: Save Results
        results_df = pd.DataFrame(all_problems)
        results_df.to_csv(f'{results_folder}problems.csv')

    @staticmethod
    @ray.remote(num_cpus=1, num_gpus=1, max_retries=10)
    def generation_run(scenario_name, seed, num_batches_per_run, batch_size, results_folder):
        RayUtils.check_storage_connection()

        try:
            run_start_time = time.time()
            run_folder = f'{results_folder}/seed_{seed}'
            os.makedirs(run_folder, exist_ok=True)
            seed_info = {
                'seed_pid': os.getpid(),
                'seed_public_ip': requests.get('https://api.ipify.org').content.decode('utf8'),
                'seed_private_ip': socket.gethostbyname(socket.gethostname()),
            }
            run_results = []
            problem_factory = ShortMissionProblemFactory(
                scenario_name=scenario_name,
                config={'seed': seed},
                verbose=0,
            )

            for batch in range(num_batches_per_run):
                batch_start_time = time.time()
                batch_folder = f'{run_folder}/batch_{batch}/'

                problems = problem_factory.generate_batch(batch_size)
                os.makedirs(batch_folder, exist_ok = True)
                problem_factory.plot_batch(batch_size, filename=f'{batch_folder}animation.gif')
                problem_factory.hindcast_planner.save_plan(batch_folder)

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
                pd.DataFrame(batch_results).to_csv(f'{batch_folder}/problems.csv')
                run_results.extend(batch_results)

                print(f'GenerationRunner: Seed {seed}/Batch {batch} finished ({time.time() - batch_start_time:.1f}s)')

            pd.DataFrame(run_results).to_csv(f'{run_folder}/problems.csv')
            
            print(f'GenerationRunner: Finished Seed {seed} ({time.time()-run_start_time:.1f}s)')

        except Exception as e:
            shutil.rmtree(run_folder, ignore_errors=True)
            print(e)
            sys.exit()

        return run_results