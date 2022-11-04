"""
    Ray Cluster Launcher can not make small changes to the nodes. For any change the cluster has
    to be completely regenerated and the whole machine newly installed. This script allows to run
    a specific shell command on all nodes. For example to install a single missing package.
"""
import datetime
import time

import pytz

from ocean_navigation_simulator.utils import cluster_utils

print(
    f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}'
)
script_start_time = time.time()


# COMMAND = 'pip install --upgrade --force-reinstall git+https://github.com/c3aidti/c3python'
# COMMAND = 'conda install -y libgcc==3.4.30'
# COMMAND = 'sudo add-apt-repository ppa:ubuntu-toolchain-r/test; sudo apt-get update; sudo apt-get install libstdc++6-4.7-dev'
# COMMAND = 'pip install ray[rllib]==1.13.0'
# COMMAND = 'pip install tensorflow'
# COMMAND = 'pip install -U ray[default]==1.13.0; pip install -U ray[rllib]==1.13.0'
# COMMAND = 'pip install --upgrade pip'
# COMMAND = 'pip install --upgrade "jax[cuda]==0.2.24" -f https://storage.googleapis.com/jax-releases/jax_releases.html'
# COMMAND = 'pip install --upgrade --force-reinstall "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
# COMMAND = 'nvcc --version'
# COMMAND = 'sudo apt-get install -y nvidia-cuda-toolkit'
# COMMAND = 'pip uninstall -y google-cloud-storage'

# COMMAND = "sudo apt-get install -y sshfs;"
# COMMAND = "sudo umount /seaweed-storage/;"
# COMMAND = "sudo mkdir -p /seaweed-storage; sudo chmod 777 /seaweed-storage;"
# COMMAND = "sshfs -o ssh_command=\"'\"ssh -i /home/ubuntu/setup/azure -o StrictHostKeyChecking=no\"'\" ubuntu@20.55.80.215:/seaweed-storage /seaweed-storage -o default_permissions"

# COMMAND = "ls -la /seaweed-storage"

# COMMAND = 'pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"'
# COMMAND = 'pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/68b5d4302c51a3ead2ffbbb972ed65fb3eb18dc7/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"'
# COMMAND = 'pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/68b5d4302c51a3ead2ffbbb972ed65fb3eb18dc7/ray-3.0.0.dev0-cp39-cp39-macosx_10_15_x86_64.whl"'

# COMMAND = 'pip install -U ray[default,rllib]==1.13.0'
# COMMAND = 'pip install -U ray[default,rllib]==1.13.0'

# COMMAND = 'rm -rf /tmp/hycom_forecast; rm -rf /tmp/hycom_hindcast/'
# COMMAND = 'pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"'
# COMMAND = 'pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/9a7aa243aa6a74e146a7ce03d3e4d51a6917d43c/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"'


# COMMAND = 'pip install gym==0.23'

# COMMAND = 'pip install - U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/75e9722a4d9c0d8d5cb0c37eb6316553f9e0789e/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"'

# The public key (. pub file) should be 644 (-rw-r--r--). The private key (id_rsa) on the client host, and the authorized_keys file on the server, should be 600 (-rw-------)
# COMMAND = 'chmod 600 ~/.ssh/azure; chmod 644 ~/.ssh/azure.pub'
# cluster_utils.run_command_on_all_nodes(COMMAND)


# COMMAND = "pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git"
# COMMAND = 'pip install "jax[cuda]==0.3.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'

# COMMAND = 'pip list | grep chex; pip list | grep jax; pip list | grep hj_reachability'

# COMMAND = 'pip install chex==0.1.5'

cluster_utils.run_command_on_all_nodes(
    "ls -la hycom_hindcast | wc; ls -la copernicus_forecast | wc", resource_group="jerome-cluster-3"
)

# cluster_utils.purge_download_temp_folders()

# cluster_utils.run_command_on_all_nodes(
#     "rm -rf /tmp/hycom_hindcast/; rm -rf /tmp/copernicus_forecast/"
# )

# cluster_utils.run_command_on_all_nodes('./OceanPlatformControl/setup/cluster-jerome/set_up_seaweed_storage.sh')


# cluster_utils.run_command_on_all_nodes('chmod ')

# cluster_utils.copy_files_to_nodes(local_dir='./setup', remote_dir='~/OceanPlatformControl/')

# import ray
# from tqdm import tqdm
# import os
# import pandas as pd
# HEAD_IP = '20.115.24.165'
# GET_PUBLIC_IP = False
# if GET_PUBLIC_IP:
#     """
#         We first get the public IPs of the nodes to simplify access. We save them in a
#         list (csv file) to use for the next time. We only ave to update the list if we connect
#         a new cluster.
#     """
#     # ray.init("ray://localhost:10001")
#
#     active_nodes = list(filter(lambda node: node['Alive'] == True, ray.nodes()))
#     nodes = []
#     public_ip = os.system(f"ssh -T -o StrictHostKeyChecking=no -i ./setup/azure ubuntu@{HEAD_IP} 'rm ~/.ssh/known_hosts'")
#     for node in tqdm(active_nodes, disable=True):
#         public_ip = os.popen(f"ssh -T -o StrictHostKeyChecking=no -i ./setup/azure ubuntu@{HEAD_IP} ssh -o StrictHostKeyChecking=no -i ./ray_bootstrap_key.pem ubuntu@{node['NodeName']} 'curl -s https://api.ipify.org'").read()
#         print(f'Public IP: {public_ip}')
#         nodes.append([node['NodeName'], public_ip])
#     nodes_df = pd.DataFrame(nodes, columns=['private_ip', 'public_ip'])
#     nodes_df.to_csv('setup/ips.csv', index=False)
# else:
#     ray.init()
#     nodes_df = pd.read_csv('setup/ips.csv')
#
#
# if 'COMMAND' in vars() or 'COMMAND' in globals():
#
#     @ray.remote(num_cpus=1)
#     def run_command_on_node(index, node, COMMAND):
#         print(f"##### Starting Node {index+1} of {nodes_df.shape[0]} on {node['public_ip']}")
#         node_start_time = time.time()
#         os.system(f"ssh -o StrictHostKeyChecking=no -i ./setup/azure ubuntu@{node['public_ip']} 'source /anaconda/etc/profile.d/conda.sh; conda activate ocean_platform; {COMMAND}'")
#         node_time = time.time()-node_start_time
#         print(f"## Node {index+1} of {nodes_df.shape[0]} finished in {node_time/60:.0f}min {node_time%60:.0f}s.")
#
#     ray.get([run_command_on_node.remote(index, node, COMMAND) for index, node in nodes_df.iterrows()])


script_time = time.time() - script_start_time
print(
    f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s."
)
