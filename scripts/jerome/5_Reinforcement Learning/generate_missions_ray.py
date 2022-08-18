import datetime
import time

from ocean_navigation_simulator.scripts.GenerationRunner import GenerationRunner
from ocean_navigation_simulator.scripts.RayUtils import RayUtils
from ocean_navigation_simulator.utils import units

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
    runs=1,
    num_batches_per_run=8,
    batch_size=4,
    problem_factory_config={
        'x_range': [units.Distance(deg=-92), units.Distance(deg=-90)],
        'y_range': [units.Distance(deg=24), units.Distance(deg=26)],
        't_range': [
            datetime.datetime(year=2022, month=4, day=1, tzinfo=datetime.timezone.utc),
            datetime.datetime(year=2022, month=4, day=10, tzinfo=datetime.timezone.utc)
        ],
        'missions_per_target': 4,
        'mission_time_range': [datetime.timedelta(hours=12), datetime.timedelta(hours=36)],
        'problem_timeout': datetime.timedelta(hours=48),
        # 'target_radius', 'goal_min_distance' have to be tuned with 'mission_time_range' for meaningful missions:
        #   - if 'goal_min_distance' is small in relation to 'target_radius' then the missions are extremely short
        #   - if
        #   0.1 deg ~= 11km = 3h @ 1m/s
        #   0.1 deg ~= 11km = 30h @ 0.1m/s
        #   1 deg ~= 111km = 30h @ 1m/s
        # if 'target_radius' is big the target is reached sooner than expected
        'problem_target_radius': 0.01,
        'target_distance_from_frame': 0.5,
        'target_min_distance': 0.15,
        # 'grid_res' has to be smaller than target_radius to prevent hj_solver errors
        'hj_planner': {
            'grid_res': 0.01,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
            'deg_around_xt_xT_box': 1.0,  # area over which to run HJ_reachability
        },
    },
    verbose=1,
)

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")