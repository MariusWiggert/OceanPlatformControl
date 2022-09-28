import json
import os
import shutil
import socket
import time
from typing import Optional

import pandas as pd
import ray
import requests
import seaborn as sns
from c3python import C3Python

from ocean_navigation_simulator.utils.units import timing

# TODO: great work! Though the architecture is a bit messy.
# I think it's better to split it up into the different functions plus move it to the utils folder.

sns.set_theme()


# 1.    ray up setup/ray-config.yaml
#       ray up --restart-only setup/ray-config.yaml
#       ray dashboard setup/ray-config.yaml
#       ray attach setup/ray-config.yaml -p 10001
# 2.    ray monitor setup/ray-config.yaml
# 3.    ray submit setup/ray-config.yaml scripts/jerome/3_Cluster_RL/evaluate_controller_ray.py

# tensorboard --logdir ~/ray_results
# ssh -L 16006:127.0.0.1:6006 olivier@my_server_ip

# How to get c3 Keyfile set up
# Step 1: generate the public and private keys locally on your computer
# in terminal run 'openssl genrsa -out c3-rsa.pem 2048' -> this generates the private key in the c3-rsa.pem file
# for public key from it run 'openssl rsa -in c3-rsa.pem -outform PEM -pubout -out public.pem'
# Step 2: move the c3-rsa.pem file to a specific folder
# Step 3: Log into C3, start jupyter service and in a cell update your users public key by
# user = c3.User.get("mariuswiggert@berkeley.edu")
# user = user.get("publicKey")
# user.publicKey = "<public key from file>"
# user.merge()

KEYFILE = "setup/c3_keys/c3-rsa.pem"
USERNAME = "mariuswiggert@berkeley.edu"


# Getting C3 Object for data downloading #
def get_c3(verbose: Optional[int] = 0):
    """Helper function to get C3 object for access to the C3 Databases.
    For now Jerome's access is hardcoded -> Need to change that!
    """
    with timing("Utils: Connect to c3 ({:.1f}s)", verbose):
        c3 = C3Python(
            url="https://dev01-seaweed-control.c3dti.ai",
            tenant="seaweed-control",
            tag="dev01",
            keyfile=KEYFILE,
            username=USERNAME,
        ).get_c3()
    return c3


# Various Utils for Using Ray and the Azure Cluster #
def init_ray(mode="cluster"):
    start = time.time()
    # Documentation: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-init
    ray.init(
        address="ray://localhost:10001" if mode == "cluster" else "auto",
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
        },
    )
    print(f"Code sent to ray nodes in {time.time() - start:.1f}s")

    active_nodes = list(filter(lambda node: node["Alive"] is True, ray.nodes()))
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


def clean_results(
    folder: str = "~/ray_results",
    filter: str = "",
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
    experiments = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if not file.startswith(".") and file.startswith(filter)
    ]
    experiments.sort(key=lambda x: os.path.getmtime(x))

    for experiment in experiments[:-ignore_most_recent] if ignore_most_recent > 0 else experiments:
        csv_file = experiment + "/progress.csv"
        if os.path.isfile(csv_file):
            with open(csv_file) as file:
                row_count = sum(1 for line in file)
                if row_count < iteration_limit:
                    if delete:
                        shutil.rmtree(experiment, ignore_errors=True)
                    if verbose > 0:
                        print(
                            f"RayUtils.clean_ray_results: Delete {experiment} with {row_count} rows"
                        )
                else:
                    if verbose > 0:
                        print(
                            f"RayUtils.clean_ray_results: Keep {experiment} with {row_count} rows"
                        )
        else:
            if delete:
                shutil.rmtree(experiment, ignore_errors=True)
            if verbose > 0:
                print(f"RayUtils.clean_ray_results: Delete {experiment} without progress.csv file")


def run_command_on_all_nodes(command, resource_group="jerome-ray-cpu"):
    vm_list = get_vm_list(resource_group)
    print("VM List fetched")

    ray.init()

    @ray.remote(num_cpus=0.1)
    def run_command_on_node(ip, command):
        print(f"##### Starting Command on Node {ip}")
        # node_start_time = time.time()
        os.system(
            f"ssh -o StrictHostKeyChecking=no -i ./setup/azure ubuntu@{ip} 'source /anaconda/etc/profile.d/conda.sh; conda activate ocean_platform; {command}'"
        )
        # node_time = time.time() - node_start_time
        # print(f"## Node {ip} finished in {node_time / 60:.0f}min {node_time % 60:.0f}s.")

    ray.get([run_command_on_node.remote(vm["publicIps"], command) for vm in vm_list])
    time.sleep(2)
    print(f'Command run on {len(vm_list)} nodes of "{resource_group}"')


def ensure_storage_connection():
    if not os.path.exists("/seaweed-storage/connected"):
        os.system("bash -i setup/set_up_seaweed_storage.sh")

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
