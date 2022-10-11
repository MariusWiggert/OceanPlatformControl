import sys
import yaml
import datetime
import time
import pytz
import os

os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'
sys.path.extend(['/home/ubuntu/OceanPlatformControl'])
print('Python %s on %s' % (sys.version, sys.platform))
print(sys.path)

from ocean_navigation_simulator.reinforcement_learning_scripts.TrainingRunner import TrainingRunner
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils

print(f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}')
script_start_time = time.time()

Utils.ray_init(logging_level="warning")

runner = TrainingRunner(
    name='big_torch_model_forecast_and_gp_error_300',
    tags=[],
    config=yaml.load(open(f'config/reinforcement_learning/training/experiment_basic.yaml'), Loader=yaml.FullLoader),
    verbose=2
).run(epochs=300)

# Utils.destroy_cluster()

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")