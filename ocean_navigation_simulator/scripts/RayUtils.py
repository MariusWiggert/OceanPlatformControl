import os
import shutil
import socket
import time
from typing import Optional
import requests
import ray
import seaborn as sns

sns.set_theme()

# 1.    ray up setup/ray-config.yaml
#       ray up --restart-only setup/ray-config.yaml
#       ray dashboard setup/ray-config.yaml
#       ray attach setup/ray-config.yaml -p 10001
# 2.    ray monitor setup/ray-config.yaml
# 3.    ray submit setup/ray-config.yaml scripts/jerome/3_Cluster_RL/evaluate_controller_ray.py

# tensorboard --logdir ~/ray_results
# ssh -L 16006:127.0.0.1:6006 olivier@my_server_ip

class RayUtils:
    @staticmethod
    def init_ray(mode='cluster'):
        start = time.time()
        # Documentation: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
        # ray.init(address='auto')
        # ray.init("ray://13.68.187.126:10001")
        ray.init(
            address='ray://localhost:10001' if mode=='cluster' else 'auto',
            runtime_env={
                'working_dir': '.',
                'excludes': ['.git', './generated_media', './ocean_navigation_simulator', './results', './scripts'],
                'py_modules': ['ocean_navigation_simulator'],
            },
        )
        print(f"Code sent to ray nodes in {time.time() - start:.1f}s")

        active_nodes = list(filter(lambda node: node['Alive'] == True, ray.nodes()))
        cpu_total = ray.cluster_resources()['CPU'] if 'CPU' in ray.cluster_resources() else 0
        gpu_total = ray.cluster_resources()['GPU'] if 'GPU' in ray.cluster_resources() else 0
        cpu_available = ray.available_resources()['CPU'] if 'CPU' in ray.available_resources() else 0
        gpu_available = ray.available_resources()['GPU'] if 'GPU' in ray.available_resources() else 0
        print(f'''This cluster consists of
    {len(active_nodes)} nodes in total
    {cpu_available}/{cpu_total} CPU resources available
    {gpu_available}/{gpu_total} GPU resources available''')

    @staticmethod
    def clean_ray_results(
        folder: str = '~/ray_results',
        filter: str = '',
        iteration_limit: Optional[int] = 2,
        delete: Optional[bool] = True,
        ignore_most_recent: Optional[int] = 1,
        verbose: Optional[int] = 0,
    ):
        """
            Ray clogs up the ~/ray_results directory by creating folders for every training start, even when
            canceling after a few iterations. This script removes all short trainings in order to simplify
            finding the important trainings in tensorboard. It however ignores the very last experiment,
            since it could be still ongoing.
        """
        experiments = [os.path.join(folder, file) for file in os.listdir(folder) if not file.startswith('.') and file.startswith(filter)]
        experiments.sort(key=lambda x: os.path.getmtime(x))

        for experiment in experiments[:-ignore_most_recent] if ignore_most_recent > 0 else experiments:
            csv_file = experiment + '/progress.csv'
            if os.path.isfile(csv_file):
                with open(csv_file) as file:
                    row_count = sum(1 for line in file)
                    if row_count < iteration_limit:
                        if delete:
                            shutil.rmtree(experiment, ignore_errors=True)
                        if verbose > 0:
                            print(f'RayUtils.clean_ray_results: Delete {csv_file} with {row_count} rows')
                    else:
                        if verbose > 0:
                            print(f'RayUtils.clean_ray_results: Keep {csv_file} with {row_count} rows')
            else:
                if delete:
                    shutil.rmtree(experiment, ignore_errors=True)
                if verbose > 0:
                    print(f'RayUtils.clean_ray_results: Delete {csv_file} without csv file')

    @staticmethod
    def check_storage_connection():
        if not os.path.exists('/seaweed-storage/connected'):
            raise FileNotFoundError(f"Seaweed Storage not connected on node {requests.get('https://api.ipify.org').content.decode('utf8')} / {socket.gethostbyname(socket.gethostname())}")