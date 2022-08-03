import ray
from tqdm import tqdm
import os
import pandas as pd

HEAD_IP = '13.68.187.126'
MACHINES = 100
DELETE_KNOWN_HOSTS = False
GET_PUBLIC_IP = True
# COMMAND = 'pip install --upgrade --force-reinstall git+https://github.com/c3aidti/c3python'
COMMAND = False

if GET_PUBLIC_IP:
    ray.init("ray://localhost:10001")
    active_nodes = list(filter(lambda node: node['Alive'] == True, ray.nodes()))
    nodes = []
    if DELETE_KNOWN_HOSTS:
        public_ip = os.system(f"ssh -T -o StrictHostKeyChecking=no -i ./setup/azure ubuntu@{HEAD_IP} 'rm ~/.ssh/known_hosts'")
    for node in tqdm(active_nodes[:MACHINES], disable=True):
        public_ip = os.popen(f"ssh -T -o StrictHostKeyChecking=no -i ./setup/azure ubuntu@{HEAD_IP} ssh -o StrictHostKeyChecking=no -i ./setup/azure ubuntu@{node['NodeName']} 'curl -s https://api.ipify.org'").read()
        print(f'Public IP: {public_ip}')
        nodes.append([node['NodeName'], public_ip])
    nodes_df = pd.DataFrame(nodes, columns=['private_ip', 'public_ip'])
    nodes_df.to_csv('setup/ips.csv', index=False)
else:
    nodes_df = pd.read_csv('setup/ips.csv')

if COMMAND:
    for index, node in tqdm(nodes_df.head(MACHINES).iterrows(), disable=True):
        print(f"##### Node {index} on {node['public_ip']}")
        os.system(f"ssh -o StrictHostKeyChecking=no -i ./setup/azure ubuntu@{node['public_ip']} 'source /anaconda/etc/profile.d/conda.sh; conda activate ocean_platform; {COMMAND}'")