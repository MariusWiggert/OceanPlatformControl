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

runner = TrainingRunner(
    # name='integrated_conv_3_3_3_fc_64_dueling_64_64_forecast_and_gp_42x42_1deg',
    # name='integrated_conv_5_5_fc_256_dueling_256_256_256_forecast_and_gp_21x21_1deg',
    # name='custom_model_256_256_256_64_64_forecast_and_gp_error',
    name='test_evaluation_tf',
    tags=[],
    config=yaml.load(open(f'config/reinforcement_learning/training/experiment_basic.yaml'), Loader=yaml.FullLoader),
    verbose=2
).run(epochs=10)

# Utils.destroy_cluster()

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")