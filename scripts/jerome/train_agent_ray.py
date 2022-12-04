import datetime
import os
import sys
import time

import pytz
import yaml

os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
sys.path.extend([os.path.expanduser("~/OceanPlatformControl")])
print("Python %s on %s" % (sys.version, sys.platform))
print(sys.path)

from ocean_navigation_simulator.reinforcement_learning.runners.TrainingRunner import (
    TrainingRunner,
)
from ocean_navigation_simulator.utils import cluster_utils

print(
    f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}'
)
script_start_time = time.time()

cluster_utils.init_ray()

runner = TrainingRunner(
    name="dense cql future",
    tags=[],
    config=yaml.load(open("config/reinforcement_learning/training.yaml"), Loader=yaml.FullLoader),
    verbose=2,
).run(epochs=2000)

# cluster_utils.destroy_cluster()

script_time = time.time() - script_start_time
print(
    f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s."
)
