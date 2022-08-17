import datetime
import time
from typing import Optional, Union, List
import numpy as np
import warnings
from matplotlib import patches
import matplotlib.pyplot as plt


from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint, SpatioTemporalPoint
from ocean_navigation_simulator.utils import units


class ShortMissionProblemFactory(ProblemFactory):
    default_config = {
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
    }
    def __init__(
        self,
        scenario_name: str,
        config: Optional[dict] = {},
        verbose: Optional[int] = 0,
    ):
        self.scenario_name = scenario_name
        self.config = ShortMissionProblemFactory.default_config | config
        self.verbose = verbose

        self.arena = ArenaFactory.create(
            scenario_name=self.scenario_name,
            x_interval=self.config['x_range'],
            y_interval=self.config['y_range'],
            t_interval=self.config['t_range'],
            verbose=self.verbose-1
        )
        self.random = np.random.default_rng(self.config['seed'] if self.config['seed'] is not None else 2022)

        self.target_x_start = self.config['x_range'][0].deg + self.config['target_distance_from_frame']
        self.target_x_end = self.config['x_range'][1].deg - self.config['target_distance_from_frame']
        self.target_y_start = self.config['y_range'][0].deg + self.config['target_distance_from_frame']
        self.target_y_end = self.config['y_range'][1].deg - self.config['target_distance_from_frame']
        self.problems = []
        self.problems_archive = []

    def has_problems_remaining(self) -> bool:
        return True

    def skips_problems(self, n):
        raise NotImplementedError

    def get_problem_list(self, limit) -> [NavigationProblem]:
        return [self.next_problem() for _ in range(limit)]

    def next_problem(self) -> NavigationProblem:
        if not self.problems:
            self.problems = self.generate_batch(batchsize=self.config['missions_per_target'])

        return self.problems.pop(0)

    def generate_batch(self, batch_size: int) -> [NavigationProblem]:
        problem_start_time = time.time()

        # Step 1: Generate Target & Starts until enough valid target/starts found
        while not (target := self.generate_target()) or not (starts := self.generate_starts(amount=batch_size)):
            pass

        # Step 2: Return Problem
        for start in starts:
            self.problems_archive.append(NavigationProblem(
                start_state=PlatformState.from_spatio_temporal_point(start),
                end_region=target.to_spatial_point(),
                target_radius=self.config['problem_target_radius'],
                timeout=self.config['problem_timeout'],
                platform_dict=self.arena.platform.platform_dict,
                optimal_time=(target.date_time - start.date_time),
                extra_info={
                    'index': len(self.problems_archive),
                    'seed': self.config['seed'],
                    'target_distance': target.distance(start),
                    'target_datetime': target.date_time.isoformat(),
                },
            ))
            if self.verbose > 1:
                print(f'ShortMissionProblemFactory: Problem created: {self.problems_archive[-1]}')

        if self.verbose > 0:
            print(f'ShortMissionProblemFactory: Batch of {batch_size} created ({time.time()-problem_start_time:.1f}s)')

        return self.problems_archive[-batch_size:]

    def generate_target(self) -> Union[PlatformState, bool]:
        """
            Generates a goal in x_range/y_range/t_range and runs a backward hj planner.
        """
        start = time.time()

        # Step 1: Generate Goal Point (x,y,t_T)
        # only sample goal times s.t. all missions will start and timeout in t_interval
        target_timestamp = self.random.integers(
            (self.config['t_range'][0] + self.config['mission_time_range'][1]).timestamp(),
            (self.config['t_range'][1] - self.config['problem_timeout'] + self.config['mission_time_range'][0]).timestamp(),
            endpoint=True
        ).item()
        target = PlatformState(
            lon=units.Distance(deg=self.target_x_start+(self.target_x_end-self.target_x_start)*self.random.random()),
            lat=units.Distance(deg=self.target_y_start+(self.target_y_end-self.target_y_start)*self.random.random()),
            date_time=datetime.datetime.fromtimestamp(target_timestamp, tz=datetime.timezone.utc)
        )
        planner_target = PlatformState(
            lon=units.Distance(deg=self.target_x_start+(self.target_x_end-self.target_x_start)*self.random.random()),
            lat=units.Distance(deg=self.target_y_start+(self.target_y_end-self.target_y_start)*self.random.random()),
            date_time=datetime.datetime.fromtimestamp(target_timestamp - self.config['mission_time_range'][1].total_seconds(), tz=datetime.timezone.utc)
        )

        # Step 2: Reject if on land
        if self.arena.is_on_land(target.to_spatial_point()):
            if self.verbose > 0:
                print(f'ShortMissionProblemFactory: Target aborted because it was on land.')
            return False

        # Step 3: Generate backward HJ Planner for this target
        self.hindcast_planner = HJReach2DPlanner(
            problem=NavigationProblem(
                start_state=planner_target,
                end_region=planner_target.to_spatial_point(),
                target_radius=self.config['problem_target_radius'],
                timeout=self.config['mission_time_range'][1],
                platform_dict=self.arena.platform.platform_dict,
            ),
            specific_settings={
                'direction': 'multi-time-reach-back',
                'n_time_vector': 199,   # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
                'accuracy': 'high',
                'artificial_dissipation_scheme': 'local_local',
                'T_goal_in_seconds': self.config['mission_time_range'][1].total_seconds(),
                'run_without_x_T': True,
            } | self.config['hj_planner'] | ({
                'x_interval': [self.config['x_range'][0].deg, self.config['x_range'][1].deg],
                'y_interval': [self.config['y_range'][0].deg, self.config['y_range'][1].deg],
            } if 'deg_around_xt_xT_box' not in self.config['hj_planner'] or not self.config['hj_planner']['deg_around_xt_xT_box'] else {}),
            verbose=self.verbose-1
        )
        # Ignore Warning that x_init might not be in reachable set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hindcast_planner.replan_if_necessary(ArenaObservation(
                platform_state=planner_target,
                true_current_at_state=self.arena.ocean_field.get_ground_truth(target.to_spatio_temporal_point()),
                forecast_data_source=self.arena.ocean_field.hindcast_data_source,
            ))

        if self.verbose > 1:
            print(f'ShortMissionProblemFactory: Target created ({time.time()-start:.1f}s) {target.to_spatio_temporal_point()}')

        return target

    def generate_starts(self, amount, silent=False) -> Union[List[SpatioTemporalPoint], bool]:
        """
            Samples from reachable goals from the already generated start.
        :return: iff there are new valid starts for the pre-existing goal.
        """
        start_time = time.time()

        points = self.hindcast_planner.sample_from_reachable_coordinates(
            random=self.random,
            t_interval=self.config['mission_time_range'],
            min_distance=self.config['target_min_distance'],
            amount=amount,
            silent=silent,
        )

        if self.verbose > 0 and not silent:
            if len(points) < amount:
                print(f'ShortMissionProblemFactory: There are only {len(points)} valid reachable points regarding mission time and minimum distance available but {amount} requested.')
            elif self.verbose > 1:
                print(f'ShortMissionProblemFactory: {len(points)} starts created ({time.time()-start_time:.1f}s)')

        return points

    def plot_batch(self,batch_size: int, filename: str, random_sample_points: Optional[int] = 10):
        plot_start_time = time.time()

        def add_drawing(ax: plt.axis):
            self.add_target_frame(ax)
            self.add_arena_frame(ax)

            # Add Starts to Plot
            for problem in self.problems_archive[-batch_size:]:
                ax.scatter(problem.start_state.lon.deg, problem.start_state.lat.deg, facecolors='none', edgecolors='r', marker='o', label='starts')

            # Plot more possible Starts
            if random_sample_points:
                for point in self.generate_starts(amount=random_sample_points, silent=True):
                    ax.scatter(point.lon.deg, point.lat.deg, facecolors='none', edgecolors='black', marker='o', label='possible sample points')

        self.hindcast_planner.plot_reachability_animation(
            filename=filename,
            add_drawing=add_drawing,
            target_min_distance=self.config['target_min_distance']
        )

        if self.verbose > 0:
            print(f'ShortMissionProblemFactory: Batch of {batch_size} plotted ({time.time()-plot_start_time:.1f}s)')

    def add_target_frame(self, ax: plt.axis) -> plt.axis:
        ax.add_patch(patches.Rectangle(
            (self.target_x_start, self.target_y_start),
            (self.target_x_end - self.target_x_start),
            (self.target_y_end - self.target_y_start),
            linewidth=2, edgecolor='g', facecolor='none', label='target sampling frame')
        )

    def add_arena_frame(self, ax: plt.axis) -> plt.axis:
        ax.add_patch(patches.Rectangle(
            (self.config['x_range'][0].deg, self.config['y_range'][0].deg),
            (self.config['x_range'][1].deg - self.config['x_range'][0].deg),
            (self.config['y_range'][1].deg - self.config['y_range'][0].deg),
            linewidth=2, edgecolor='r', facecolor='none', label='arena frame')
        )