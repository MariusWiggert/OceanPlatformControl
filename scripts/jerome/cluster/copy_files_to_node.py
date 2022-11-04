import os
import sys

import ray


os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
sys.path.extend(["/home/ubuntu/OceanPlatformControl"])
print("Python %s on %s" % (sys.version, sys.platform))
print(sys.path)

from ocean_navigation_simulator.utils import cluster_utils


@ray.remote(resources={"Node": 1})
def download_all_files(i):
    """purge download temp folders"""

    cluster_utils.ensure_storage_connection()

    os.system("mkdir -p /home/ubuntu/hycom_hindcast; mkdir -p /home/ubuntu/copernicus_forecast")
    os.system(
        "rsync /seaweed-storage/hycom_hindcast/* /home/ubuntu/hycom_hindcast/ -v; rsync /seaweed-storage/copernicus_forecast/* /home/ubuntu/copernicus_forecast/ -v"
    )

    print(f"Finished Node {i}")


nodes, _, _ = cluster_utils.init_ray()

ray.get([download_all_files.remote(i) for i in range(len(nodes))])

print("Finished")
