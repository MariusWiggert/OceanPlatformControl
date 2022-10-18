import json
import os
import socket
from typing import Optional

import pandas as pd
import requests
import ray
from ocean_navigation_simulator.utils.misc import timing


# 1.    ray up setup/ray-config.yaml
#       ray up --restart-only setup/ray-config.yaml
#       ray dashboard setup/ray-config.yaml
#       ray attach setup/ray-config.yaml -p 10001
# 2.    ray monitor setup/ray-config.yaml
# 3.    ray submit setup/ray-config.yaml scripts/jerome/3_Cluster_RL/evaluate_controller_ray.py

# tensorboard --logdir ~/ray_results
# ssh -L 16006:127.0.0.1:6006 olivier@my_server_ip


### Various Utils for Using Ray and the Azure Cluster ###
def init_ray(**kwargs):
    """
    Initialises ray with a runtime environment.
    Ray then sends the specified code directories to all cluster nodes.
    Args:
        **kwargs: to be passed to ray.init()
    """
    with timing("Code sent to ray nodes in {:.1f}s", verbose=1):
        # Documentation:
        # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
        # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-init
        ray.init(
            runtime_env={
                "working_dir": ".",
                "excludes": [
                    ".git",
                    "./generated_media",
                    "./ocean_navigation_simulator",
                    "./results",
                    "./scripts",
                ],
                "py_modules": ["ocean_navigation_simulator"],
                "env_vars": {"LOGLEVEL": "WARN"},
            },
            **kwargs,
        )

    print_ray_resources()


def print_ray_resources():
    """
    Prints available nodes/cpus/gpus in the ray instance.
    ray.init() has to be called first.
    """
    active_nodes = list(filter(lambda node: node["Alive"] == True, ray.nodes()))
    cpu_total = ray.cluster_resources()["CPU"] if "CPU" in ray.cluster_resources() else 0
    gpu_total = ray.cluster_resources()["GPU"] if "GPU" in ray.cluster_resources() else 0
    cpu_available = ray.available_resources()["CPU"] if "CPU" in ray.available_resources() else 0
    gpu_available = ray.available_resources()["GPU"] if "GPU" in ray.available_resources() else 0

    print(
        f"""This cluster consists of
    {len(active_nodes)} nodes in total
    {cpu_available}/{cpu_total} CPU resources available
    {gpu_available}/{gpu_total} GPU resources available"""
    )


def destroy_cluster():
    """
    Shuts down the ray cluster. NOT WORKING YET.
    """
    os.system("ray down -y setup/ray-config.yaml")


def run_command_on_all_nodes(command, resource_group="jerome-ray-cpu"):
    """
    Runs a command on all machines in the specified resourcegroup of our Azure Directory.
    This allows to quickly install new dependencies without running
    the whole installation script.
    Example:
        ray.init()
        run_command_on_all_nodes('ls -la', 'your-resource-group')
    :arg
        command: the console command to run
        resource-group: the resource group where the machines should be found
    """
    vm_list = get_vm_list(resource_group)
    print(f"VM List fetched")

    ray.init()

    @ray.remote(num_cpus=0.1)
    def run_command_on_node(ip, command):
        with timing(f"## Node {ip} finished in {{:.0f}}s."):
            print(f"##### Starting Command ({command}) on Node {ip}")
            os.system(
                f"ssh -o StrictHostKeyChecking=no -i ~/.ssh/azure ubuntu@{ip} 'source /anaconda/etc/profile.d/conda.sh; conda activate ocean_platform; {command}'"
            )

    ray.get([run_command_on_node.remote(vm["publicIps"], command) for vm in vm_list])
    print(f'Command run on {len(vm_list)} nodes of "{resource_group}"')


def copy_files_to_nodes(local_dir, remote_dir, resource_group="jerome-ray-cpu"):
    """
    Runs a command on all machines in the specified resourcegroup of our Azure Directory.
    This allows to quickly install new dependencies without running
    the whole installation script.
    Example:
        ray.init()
        run_command_on_all_nodes('ls -la', 'your-resource-group')
    :arg
        command: the console command to run
        resource-group: the resource group where the machines should be found
    """
    vm_list = get_vm_list(resource_group)
    print(f"VM List fetched")

    ray.init()

    @ray.remote(num_cpus=0.1)
    def run_command_on_node(ip, local_dir, remote_dir):
        with timing(f"## Node {ip} finished in {{:.0f}}s."):
            print(f"##### Copying directory ({local_dir}) to Node {ip}")
            os.system(
                f"scp -r -o StrictHostKeyChecking=no -i ~/.ssh/azure {local_dir} ubuntu@{ip}:{remote_dir}"
            )

    ray.get([run_command_on_node.remote(vm["publicIps"], local_dir, remote_dir) for vm in vm_list])
    print(f'Command run on {len(vm_list)} nodes of "{resource_group}"')


def ensure_storage_connection():
    """
    Checks if the Azure storage is connected.
    Tries to reconnect and throws an error if not possible.
    """
    if not os.path.exists("/seaweed-storage/connected"):
        os.system("bash -i setup/cluster-jerome/set_up_seaweed_storage.sh")

    if not os.path.exists("/seaweed-storage/connected"):
        raise FileNotFoundError(
            f"Seaweed Storage not connected on node {requests.get('https://api.ipify.org').content.decode('utf8')} / {socket.gethostbyname(socket.gethostname())}"
        )


def get_vm_list(resource_group: Optional[str] = "jerome-ray-cpu"):
    return json.loads(
        os.popen(f"az vm list --resource-group '{resource_group}' --show-details").read()
    )


def print_vm_table(vm_dict: dict = None):
    vm_df = pd.DataFrame(
        [
            {
                "name": vm["name"],
                "vmSize": vm["hardwareProfile"]["vmSize"],
                "status": vm["powerState"],
                "public_ip": vm["publicIps"],
                "private_ip": vm["privateIps"],
            }
            for vm in (vm_dict if vm_dict is not None else get_vm_list())
        ]
    )
    with pd.option_context(
        "display.width", 500, "display.max_columns", 100, "display.max_colwidth", 100
    ):
        print(vm_df)
