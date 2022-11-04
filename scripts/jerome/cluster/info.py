import os
import sys

# import ray


os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
sys.path.extend(["/home/ubuntu/OceanPlatformControl"])
print("Python %s on %s" % (sys.version, sys.platform))
print(sys.path)

from ocean_navigation_simulator.utils import cluster_utils

cluster_utils.init_ray()
