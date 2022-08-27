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
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatioTemporalPoint
from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory
from ocean_navigation_simulator.utils import units


class ShortMissionProblemFactory:
    def __init__(
        self,
        scenario_name: str,
        config: Optional[dict] = {},
        verbose: Optional[int] = 0,
    ):
        self.scenario_name = scenario_name
        self.config = config
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
        self.target_x_end   = self.config['x_range'][1].deg - self.config['target_distance_from_frame']
        self.target_y_start = self.config['y_range'][0].deg + self.config['target_distance_from_frame']
        self.target_y_end   = self.config['y_range'][1].deg - self.config['target_distance_from_frame']
        # only sample goal times s.t. all missions will start and timeout in t_interval
        self.target_t_start = (self.config['t_range'][0] + self.config['problem_timeout']).timestamp()
        self.target_t_end   = self.config['t_range'][1].timestamp(),

        self.starts_x_start = self.config['x_range'][0].deg + self.config['start_distance_from_frame']
        self.starts_x_end = self.config['x_range'][1].deg - self.config['start_distance_from_frame']
        self.starts_y_start = self.config['y_range'][0].deg + self.config['start_distance_from_frame']
        self.starts_y_end = self.config['y_range'][1].deg - self.config['start_distance_from_frame']

        self.planner_config = {
            'direction': 'multi-time-reach-back',
            'n_time_vector': 199,   # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
            'accuracy': 'high',
            'artificial_dissipation_scheme': 'local_local',
            'run_without_x_T': True,
        } | self.config['hj_planner'] | ({
            'x_interval': [self.config['x_range'][0].deg, self.config['x_range'][1].deg],
            'y_interval': [self.config['y_range'][0].deg, self.config['y_range'][1].deg],
        } if 'deg_around_xt_xT_box' not in self.config['hj_planner'] or not self.config['hj_planner']['deg_around_xt_xT_box'] else {})

        self.problems = []
        self.problems_archive = []

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
                timeout=(target.date_time - start.date_time),
                platform_dict=self.arena.platform.platform_dict,
                extra_info={
                    'optimal_time_in_h': self.hindcast_planner.interpolate_value_function_in_hours(point=start).item(),
                    'target_distance_in_deg': target.distance(start),
                    'timeout_datetime': target.date_time.isoformat(),
                    'factory_seed': self.config['seed'],
                    'factory_index': len(self.problems_archive),
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
        # Planner starts from timeout backwards (this is a trick, so we can use the planner after max_mission_range)!
        target = PlatformState(
            lon=units.Distance(deg=self.target_x_start+(self.target_x_end-self.target_x_start)*self.random.random()),
            lat=units.Distance(deg=self.target_y_start+(self.target_y_end-self.target_y_start)*self.random.random()),
            date_time=datetime.datetime.fromtimestamp(self.random.integers(self.target_t_start, self.target_t_end).item(), tz=datetime.timezone.utc)
        )
        start_state = PlatformState(
            lon=target.lon,
            lat=target.lat,
            date_time=target.date_time-self.config['problem_timeout']
        )

        # Step 2: Reject if on land
        if self.arena.is_on_land(target.to_spatial_point()):
            if self.verbose > 0:
                print(f'ShortMissionProblemFactory: Target aborted because it was on land.')
            return False

        # Step 3: Generate backward HJ Planner for this target
        self.hindcast_planner = HJReach2DPlanner(
            problem=NavigationProblem(
                start_state=start_state,
                end_region=target.to_spatial_point(),
                target_radius=self.config['problem_target_radius'],
                timeout=self.config['problem_timeout'],
                platform_dict=self.arena.platform.platform_dict,
            ),
            specific_settings=self.planner_config,
            verbose=self.verbose-1
        )
        # Ignore Warning that x_init might not be in reachable set
        # with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        self.hindcast_planner.replan_if_necessary(ArenaObservation(
            platform_state=start_state,
            true_current_at_state=self.arena.ocean_field.get_ground_truth(target.to_spatio_temporal_point()),
            forecast_data_source=self.arena.ocean_field.hindcast_data_source,
        ))
        # Update target time to maximum of planner
        target.date_time = datetime.datetime.fromtimestamp(int(self.hindcast_planner.current_data_t_0+self.hindcast_planner.reach_times[-1]), tz=datetime.timezone.utc)

        if self.verbose > 1:
            print(f'ShortMissionProblemFactory: Target created ({time.time()-start:.1f}s) {target.to_spatio_temporal_point()}')

        return target

    def generate_starts(self, amount, silent=False) -> Union[List[SpatioTemporalPoint], bool]:
        """
            Samples from reachable goals from the already generated start.
        :return: iff there are new valid starts for the pre-existing goal.
        """
        start_time = time.time()

        # we start missions at earliest
        points = self.hindcast_planner.sample_from_reachable_coordinates(
            random=self.random,
            reach_interval=[
                (self.config['mission_time_range'][0]).total_seconds(),
                (self.config['mission_time_range'][1]).total_seconds(),
            ],
            frame_interval=[
                [self.starts_x_start, self.starts_x_end],
                [self.starts_y_start, self.starts_y_end],
            ],
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

    def plot_batch(self, batch_size: int, filename: str, random_sample_points: Optional[int] = 10):
        plot_start_time = time.time()

        def add_drawing(ax: plt.axis, rel_time_in_seconds):
            self.arena.plot_arena_frame_on_map(ax)
            self.add_target_frame(ax)
            self.add_starts_frame(ax)

            # Add Starts to Plot
            for problem in self.problems_archive[-batch_size:]:
                ax.scatter(problem.start_state.lon.deg, problem.start_state.lat.deg, facecolors='none', edgecolors='r', marker='o', label='starts')

            # Plot more possible Starts
            if random_sample_points:
                for point in self.generate_starts(amount=random_sample_points, silent=True):
                    ax.scatter(point.lon.deg, point.lat.deg, facecolors='none', edgecolors='black', marker='o', label='possible sample points')

            ax.set_title(f"Multi-Reach at time ({rel_time_in_seconds/3600:.1f}h)")

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
            linewidth=4, edgecolor='g', facecolor='none', label='target sampling frame')
        )

    def add_starts_frame(self, ax: plt.axis) -> plt.axis:
        ax.add_patch(patches.Rectangle(
            (self.starts_x_start, self.starts_y_start),
            (self.starts_x_end - self.starts_x_start),
            (self.starts_y_end - self.starts_y_start),
            linewidth=4, edgecolor='g', facecolor='none', label='start sampling frame')
        )

    def run_forecast(self, batch_folder):
        """
            We have to trick the planner here to run exactly the forecasts we need:
                - we take any problem of the current batch (they share the same target, so the planner is the same)
                - we run until timeout (problem.is_done == -1)
                - we reset the platform x&y to start coordinates s.t. we never runs out of area
        """
        problem_factory = FileMissionProblemFactory(csv_file=f'{batch_folder}/problems.csv')
        problem = problem_factory.next_problem()

        arena = ArenaFactory.create(scenario_name=self.scenario_name, problem=problem)

        forecast_planner = HJReach2DPlanner(problem=problem, specific_settings=self.hindcast_planner.specific_settings | {
            'planner_path': batch_folder,
            'save_after_planning': True
        }, verbose=self.verbose-1)

        observation = arena.reset(problem.start_state)
        while problem.is_done(observation.platform_state) != -1:
            observation = arena.step(forecast_planner.get_action(observation))
            arena.platform.state.lon = problem.start_state.lon
            arena.platform.state.lat = problem.start_state.lat