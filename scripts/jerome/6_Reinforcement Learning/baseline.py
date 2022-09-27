import datetime
import time
import pytz

from ocean_navigation_simulator.reinforcement_learning.scripts.BaselineRunner import BaselineRunner

print(
    f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S")}'
)
script_start_time = time.time()


runner = BaselineRunner(
    generation_folder="/seaweed-storage/generation/5_fixed_planner_dates_nightrun_2022_08_20_05_31_57/",
    ray_options={
        "max_retries": 10,
        "resources": {
            "CPU": 1.0,
            "GPU": 0.0,
            "RAM": 4000,
            "Worker CPU": 1.0,
        },
    },
    verbose=10,
)


script_time = time.time() - script_start_time
print(
    f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s."
)
