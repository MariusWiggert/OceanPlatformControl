import datetime

from ocean_navigation_simulator.problem_factories.ShortMissionProblemFactory import ShortMissionProblemFactory
from ocean_navigation_simulator.utils import units

for i in range(1):
    problem_factory = ShortMissionProblemFactory(
        scenario_name='gulf_of_mexico_HYCOM_hindcast',
        config={
            'seed': i,
            'x_range': [units.Distance(deg=-92.5), units.Distance(deg=-89.5)],
            'y_range': [units.Distance(deg=23.5), units.Distance(deg=26.5)],
            't_range': [
                datetime.datetime(year=2022, month=4, day=1, tzinfo=datetime.timezone.utc),
                datetime.datetime(year=2022, month=4, day=11, tzinfo=datetime.timezone.utc)
            ],
            'missions_per_target': 4,
            'mission_time_range': [datetime.timedelta(hours=24), datetime.timedelta(hours=36)],
            'problem_timeout': datetime.timedelta(hours=48),
            # 'target_radius', 'goal_min_distance' have to be tuned with 'mission_time_range' for meaningful missions:
            #   - if 'goal_min_distance' is small in relation to 'target_radius' then the missions are extremely short
            #   - if
            #   0.1 deg ~= 11km = 3h @ 1m/s
            #   0.1 deg ~= 11km = 30h @ 0.1m/s
            #   1 deg ~= 111km = 30h @ 1m/s
            # if 'target_radius' is big the target is reached sooner than expected
            'problem_target_radius': 0.05,
            'start_distance_from_frame': 0.5,
            'target_distance_from_frame': 1.0,
            'target_min_distance': 0.15,
            # 'grid_res' has to be smaller than target_radius to prevent hj_solver errors
            'hj_planner': {
                'grid_res': 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
                'deg_around_xt_xT_box': 2.0,  # area over which to run HJ_reachability
            },
        },
        verbose=1,
    )
    problems = problem_factory.generate_batch(4)
    problem_factory.plot_batch(4, filename=f'/seaweed-storage/tmp/animation{i}.gif')
    # problem_factory.hindcast_planner.save_plan('/seaweed-storage/tmp/planner/')

