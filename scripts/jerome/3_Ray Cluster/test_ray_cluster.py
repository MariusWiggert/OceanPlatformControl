"""
    This test script can be run on a newly generated cluster to check if it works. The processes
    return their IP + PID. This helps to see if all the nodes work and if the work is balanced.
"""
from collections import Counter
import socket
import time
import ray
from requests import get
import os

print('Script started ...')
script_start_time = time.time()

# ray.init(address='auto')
# ray.init("ray://13.68.187.126:10001")
ray.init("ray://localhost:10001")

# 1.    ray up setup/ray-config-gpu.yaml
#       ray dashboard setup/ray-config-gpu.yaml
#       ray attach setup/ray-config-gpu.yaml -p 10001
# 2.    ray monitor setup/ray-config-gpu.yaml
# 3.    ray submit setup/ray-config-gpu.yaml scripts/jerome/3_Cluster_RL/evaluate_controller_ray.py

active_nodes = list(filter(lambda node: node['Alive'] == True, ray.nodes()))
cpu_resources = ray.cluster_resources()['CPU'] if 'CPU' in ray.cluster_resources() else 0
gpu_resources = ray.cluster_resources()['GPU'] if 'GPU' in ray.cluster_resources() else 0
print(f'''This cluster consists of
    {len(active_nodes):.0f} nodes
    {cpu_resources:.0f} CPU resources
    {gpu_resources:.0f} GPU resources''')

@ray.remote(num_cpus=1)
def f():
    public_ip = get('https://api.ipify.org').content.decode('utf8')
    private_ip = socket.gethostbyname(socket.gethostname())
    pid = os.getpid()

    time.sleep(1)

    print(f'Task Finished ({public_ip}, {private_ip}, {pid})')

    return f'{public_ip}, {private_ip}, {pid}'

ip_addresses = ray.get([f.remote() for _ in range(1000)])

for ip_address, num_tasks in sorted(Counter(ip_addresses).items(), key=lambda pair: (pair[0], pair[1]), reverse=True):
    print('    {} tasks on {}'.format(num_tasks, ip_address))

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")