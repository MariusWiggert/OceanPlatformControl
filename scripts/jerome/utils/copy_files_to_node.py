import os

import ray

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.utils import cluster_utils

@ray.remote(resources={"Node": 1})
def download_all_files(i):
    cluster_utils. ensure_storage_connection()
    "purge download temp folders"
    os.system("rm -rf /home/ubuntu/hycom_hindcast; rm -rf /home/ubuntu/copernicus_forecast")
    os.system("cp -r /seaweed-storage/hycom_hindcast /home/ubuntu; cp -r /seaweed-storage/copernicus_forecast /home/ubuntu")

    print(f"Finished Node {i}")

nodes, _, _ = cluster_utils.init_ray()

ray.get([download_all_files.remote(i) for i in range(len(nodes))])

print("Finished")
