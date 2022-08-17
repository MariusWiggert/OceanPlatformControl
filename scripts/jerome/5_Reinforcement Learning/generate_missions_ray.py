import time

from ocean_navigation_simulator.scripts.GenerationRunner import GenerationRunner
from ocean_navigation_simulator.scripts.RayUtils import RayUtils

print('Script started ...')
script_start_time = time.time()

RayUtils.init_ray()

runner = GenerationRunner(
    name='test',
    scenario_name='gulf_of_mexico_HYCOM_hindcast',
    # 1 run of 8 batches @ 4 starts: ~8min
    # ~ 7.5 runs / hour / machines => 60 runs / hour @ 8 machines @ 8 batches @ 4 starts
    # 360 runs ~ 6h
    # 200 runs ~ 3.3h
    runs=200,
    num_batches_per_run=8,
    batch_size=4,
)

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")