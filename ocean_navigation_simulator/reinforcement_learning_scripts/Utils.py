import contextlib
import json
import os
import shutil
import socket
import time
from types import SimpleNamespace
from typing import Optional
import pandas as pd
import psutil
import pynvml
import requests
import ray
import seaborn as sns
from c3python import C3Python

from ocean_navigation_simulator.utils.bcolors import bcolors

sns.set_theme()

# 1.    ray up setup/ray-config.yaml
#       ray up --restart-only setup/ray-config.yaml
#       ray dashboard setup/ray-config.yaml
#       ray attach setup/ray-config.yaml -p 10001
# 2.    ray monitor setup/ray-config.yaml
# 3.    ray submit setup/ray-config.yaml scripts/jerome/3_Cluster_RL/evaluate_controller_ray.py

# tensorboard --logdir ~/ray_results
# ssh -L 16006:127.0.0.1:6006 olivier@my_server_ip

class Utils:
    @staticmethod
    def ray_init(mode='cluster'):
        start = time.time()
        # Documentation: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
        # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-init
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
    def destroy_cluster():
        os.system('ray down -y setup/ray-config.yaml')

    @staticmethod
    def clean_results(
        folder: str = '~/ray_results',
        filter: str = '',
        iteration_limit: Optional[int] = 10,
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
                            print(f'RayUtils.clean_ray_results: Delete {bcolors.FAIL}{experiment} with {row_count} rows {bcolors.ENDC}')
                    else:
                        if verbose > 0:
                            print(f'RayUtils.clean_ray_results: Keep   {bcolors.OKGREEN}{experiment} with {row_count} rows {bcolors.ENDC}')
            else:
                if delete:
                    shutil.rmtree(experiment, ignore_errors=True)
                if verbose > 0:
                    print(f'RayUtils.clean_ray_results: Delete {bcolors.FAIL}{experiment} without progress.csv file {bcolors.ENDC}')

    @staticmethod
    def run_command_on_all_nodes(command, resource_group='jerome-ray-cpu'):
        vm_list = Utils.get_vm_list(resource_group)
        print(f'VM List fetched')

        ray.init()
        @ray.remote(num_cpus=0.1)
        def run_command_on_node(ip, command):
            print(f"##### Starting Command on Node {ip}")
            node_start_time = time.time()
            os.system(f"ssh -o StrictHostKeyChecking=no -i ./setup/keys/azure ubuntu@{ip} 'source /anaconda/etc/profile.d/conda.sh; conda activate ocean_platform; {command}'")
            node_time = time.time() - node_start_time
            # print(f"## Node {ip} finished in {node_time / 60:.0f}min {node_time % 60:.0f}s.")

        ray.get([run_command_on_node.remote(vm['publicIps'], command) for vm in vm_list])
        time.sleep(2)
        print(f'Command run on {len(vm_list)} nodes of "{resource_group}"')

    @staticmethod
    def ensure_storage_connection():
        if not os.path.exists('/seaweed-storage/connected'):
            os.system('bash setup/set_up_seaweed_storage.sh')

        if not os.path.exists('/seaweed-storage/connected'):
            raise FileNotFoundError(f"Seaweed Storage not connected on node {requests.get('https://api.ipify.org').content.decode('utf8')} / {socket.gethostbyname(socket.gethostname())}")

    @staticmethod
    def get_c3(verbose: Optional[int] = 0):
        if not hasattr(Utils, '_c3'):
            with Utils.timing('Utils: Connect to c3 ({:.1f}s)', verbose):
                Utils._c3 = C3Python(
                    url='https://dev01-seaweed-control.c3dti.ai',
                    tenant='seaweed-control',
                    tag='dev01',
                    keyfile='setup/keys/c3-rsa-jerome',
                    username='jeanninj@berkeley.edu',
                ).get_c3()
        return Utils._c3

    @staticmethod
    @contextlib.contextmanager
    def timing(string, verbose: Optional[int] = 0):
        if verbose > 0:
            start = time.time()
        yield
        if verbose > 0:
            print(string.format(time.time()-start))

    @staticmethod
    def get_vm_list(resource_group: Optional[str] = 'jerome-ray-cpu'):
        return json.loads(os.popen(f"az vm list --resource-group '{resource_group}' --show-details").read())

    @staticmethod
    def print_vm_table(vm_dict: dict = None):
        vm_df = pd.DataFrame([{
            'name': vm['name'],
            'vmSize': vm['hardwareProfile']['vmSize'],
            'status': vm['powerState'],
            'public_ip': vm['publicIps'],
            'private_ip': vm['privateIps'],
        } for vm in (vm_dict if vm_dict is not None else Utils.get_vm_list())])
        with pd.option_context('display.width', 500, 'display.max_columns', 100, 'display.max_colwidth', 100):
            print(vm_df)

    @staticmethod
    def get_process_information_dict() -> dict:
        try:
            pynvml.nvmlInit()
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception as e:
            gpu_info = SimpleNamespace(total=0, free=0, used=0)

        return {
            'process_pid': os.getpid(),
            'process_public_ip': requests.get('https://api.ipify.org').content.decode('utf8'),
            'process_private_ip': socket.gethostbyname(socket.gethostname()),
            'process_ram': f'{psutil.Process().memory_info().rss / 1e6:.1f}MB',
            'process_gpu_total': f'{gpu_info.total / 1e6:,.0f}MB',
            'process_gpu_used': f'{gpu_info.free / 1e6:,.0f}MB',
            'process_gpu_free': f'{gpu_info.used / 1e6:,.0f}MB',
        }
