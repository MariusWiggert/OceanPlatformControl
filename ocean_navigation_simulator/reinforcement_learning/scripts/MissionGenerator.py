import datetime
import time
import warnings
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.ArenaFactory import (
    ArenaFactory,
    MissingOceanFileException,
    CorruptedOceanFileException,
)
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.problem_factories.FileProblemFactory import (
    FileProblemFactory,
)
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import bcolors, timing


class MissionGenerator:
    """A flexible class to generate missions"""

    def __init__(
        self,
        scenario_file: str,
        config: Optional[dict] = {},
        verbose: Optional[int] = 0,
    ):
        self.scenario_file = scenario_file
        self.config = config
        self.verbose = verbose

        self.random = np.random.default_rng(self.config["seed"])


    def generate_batch(self) -> [NavigationProblem]:
        """ generate a target and a batch of starts according to config """
        self.problems = []

        with timing(
            f"MissionGenerator: Batch created ({{:.1f}}s)", self.verbose - 1
        ):
            # Step 1: Generate Target & Starts until enough valid target/starts found
            while not (target := self.generate_target()) or not (starts := self.generate_starts()):
                pass

            # Step 2: Create Problems
            for start in starts:
                self.problems.append(
                    NavigationProblem(
                        start_state=PlatformState.from_spatio_temporal_point(start),
                        end_region=target.to_spatial_point(),
                        target_radius=self.config["problem_target_radius"],
                        timeout=(target.date_time - start.date_time),
                        platform_dict=self.arena.platform.platform_dict,
                        extra_info={
                            "timeout_datetime": target.date_time.isoformat(),

                            # Distances
                            "start_target_distance": target.haversine(start),
                            "start_distance_to_shore": self.arena.ocean_field.hindcast_data_source.distance_to_land(start),
                            "target_distance_to_shore": '',

                            # Feasability

                            # Mission Time
                            "optimal_time_in_h": self.hindcast_planner.interpolate_value_function_in_hours(
                                point=start
                            ).item(),

                            # Planner Caching
                            "x_cache": '',
                            "y_cache": '',
                            "t_cache": '',

                            # Factory
                            "factory_seed": self.config["seed"],
                            "factory_index": len(self.problems),
                        },
                    )
                )
                if self.verbose > 1:
                    print(f"MissionGenerator: Problem created: {self.problems[-1]}")

        return self.problems

    def generate_target(self) -> Union[PlatformState, bool]:
        """
        Samples a target in x_range/y_range/t_range.
        - ensures that there are enough healthy files available in t_interval
        - ensures distance to shore
        - runs hj planner for target
        """
        start = time.time()

        # Step 1: Generate Target (x,y,t_T)
        # Planner starts from timeout backwards (this is a trick, so we can use the planner after max_mission_range)!
        target = PlatformState(
            lon=units.Distance(
                deg=self.random.uniform(
                    self.config["x_range"][0].deg, self.config["x_range"][1].deg
                )
            ),
            lat=units.Distance(
                deg=self.random.uniform(
                    self.config["y_range"][0].deg, self.config["y_range"][1].deg
                )
            ),
            # only sample goal times s.t. all missions will start and timeout in t_interval
            date_time=datetime.datetime.fromtimestamp(
                self.random.uniform(
                    (self.config["t_range"][0]+self.config["problem_timeout"]).timestamp(),
                    self.config["t_range"][1].timestamp()
                ).item(),
                tz=datetime.timezone.utc,
            ),
        )
        start_state = PlatformState(
            lon=target.lon,
            lat=target.lat,
            date_time=target.date_time - self.config["problem_timeout"],
        )

        # Step 2: Reject if files are missing or corrupted
        try:
            self.arena = ArenaFactory.create(
                scenario_file=self.scenario_file,
                x_interval=self.config["x_range"],
                y_interval=self.config["y_range"],
                t_interval=[start_state.date_time, target.date_time],
                verbose=self.verbose-2,
            )
        except MissingOceanFileException as e:
            if self.verbose > 0:
                print(
                    f"MissionGenerator: {bcolors.FAIL}Target aborted because of missing files ({time.time()-start:.1f}s).{bcolors.ENDC}"
                )
            if self.verbose > 1:
                print(e)
            return False
        except CorruptedOceanFileException as e:
            if self.verbose > 0:
                print(
                    f"MissionGenerator: {bcolors.FAIL}Target aborted because of corrupted file: ({time.time()-start:.1f}s).{bcolors.ENDC}"
                )
            if self.verbose > 1:
                print(e)
            return False

        # Step 2: Reject if to close to land
        distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
            target.to_spatial_point()
        )

        if distance_to_shore.deg < self.config["target_distance_from_land"]:
            if self.verbose > 0:
                print(
                    f"MissionGenerator: {bcolors.FAIL}Target {target.to_spatial_point()} aborted because too close to land: {distance_to_shore}  ({time.time()-start:.1f}s).{bcolors.ENDC}"
                )
            return False

        # Step 3: Run multi-time-back HJ Planner
        self.hindcast_planner = HJReach2DPlanner(
            problem=NavigationProblem(
                start_state=start_state,
                end_region=target.to_spatial_point(),
                target_radius=self.config['problem_target_radius'],
                timeout=self.config['problem_timeout'],
                platform_dict=self.arena.platform.platform_dict,
            ),
            specific_settings={
                'direction': 'multi-time-reach-back',
                'n_time_vector': 199,   # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
                'accuracy': 'high',
                'artificial_dissipation_scheme': 'local_local',
                'run_without_x_T': True,
            } | self.config['hj_planner'] | ({
                'x_interval': [self.config['x_range'][0].deg, self.config['x_range'][1].deg],
                'y_interval': [self.config['y_range'][0].deg, self.config['y_range'][1].deg],
            } if 'deg_around_xt_xT_box' not in self.config['hj_planner'] or not self.config['hj_planner']['deg_around_xt_xT_box'] else {}),
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

        if self.verbose > 0:
            print(
                f"MissionGenerator: {bcolors.OKGREEN}Target created: {target.to_spatio_temporal_point()} ({time.time()-start:.1f}s).{bcolors.ENDC}"
            )

        return target

    def generate_starts(self) -> Union[List[SpatioTemporalPoint], bool]:
        """
            Samples from reachable starts from the already generated target.
        :return: List of valid starts for the pre-existing target.
        """
        start_time = time.time()

        # we start missions at earliest
        feasible_points = self.sample_feasible_points(
            frame_interval=[
                [self.starts_x_start, self.starts_x_end],
                [self.starts_y_start, self.starts_y_end],
            ],
            min_distance=self.config["target_min_distance"],
            amount=amount,
        )

        infeasible_points = self.sample_infeasible_points()

        if self.verbose > 0:
            if len(feasible_points) < amount:
                print(
                    f"MissionGenerator: There are only {len(feasible_points)} valid reachable points regarding mission time and minimum distance available but {amount} requested."
                )
            elif len(infeasible_points) < amount:
                print(
                    f"MissionGenerator: There are only {len(infeasible_points)} valid reachable points regarding mission time and minimum distance available but {amount} requested."
                )
            elif self.verbose > 1:
                print(
                    f"MissionGenerator: {len(feasible_points + infeasible_points)} starts created ({time.time()-start_time:.1f}s)"
                )

        return feasible_points + infeasible_points

    def sample_feasible_points(
        self,
        frame_interval: List[List[float]],
        min_distance: Optional[float] = 0,
        amount: Optional[int] = 1,
    ) -> List[SpatioTemporalPoint]:
        planner = self.hindcast_planner

        # Step 1: Find reachable points with minimum distance
        all_values_dim = (planner.all_values[0] - planner.all_values.min()) * (
            planner.current_data_t_T - planner.current_data_t_0
        )
        reachable_condition = (
            self.config["feasible_mission_time"][0].total_seconds() < all_values_dim & (
                all_values_dim < self.config["feasible_mission_time"][1].total_seconds()
            )
        )
        min_distance_condition = planner.initial_values > (
            min_distance / planner.characteristic_vec[0]
        )
        frame_condition_x = (frame_interval[0][0] < planner.grid.states[:, :, 0]) & (
            planner.grid.states[:, :, 0] < frame_interval[0][1]
        )
        frame_condition_y = (frame_interval[1][0] < planner.grid.states[:, :, 1]) & (
            planner.grid.states[:, :, 1] < frame_interval[1][1]
        )
        points_to_sample = np.argwhere(
            reachable_condition & min_distance_condition & frame_condition_x & frame_condition_y
        )

        if self.verbose > 1:
            print(f'Natural Feasability Ratio: {points_to_sample.shape[0] / planner.all_values.shape[0]}')

        if self.verbose > 2:
            print(
                "HJPlannerBase: reach_time_condition  =", np.argwhere(reachable_condition).shape[0]
            )
            print(
                "HJPlannerBase: distance_condition =", np.argwhere(min_distance_condition).shape[0]
            )
            print("HJPlannerBase: points_to_sample   =", points_to_sample.shape[0])

        # Step 2: Return List of SpatioTemporalPoint
        sampled_points = []
        for _ in range(min(amount, points_to_sample.shape[0])):
            sample_index = self.random.integers(points_to_sample.shape[0])
            sampled_point = points_to_sample[sample_index]
            points_to_sample = np.delete(points_to_sample, sample_index, axis=0)
            coordinates = planner.grid.states[sampled_point[0], sampled_point[1], :]

            # Add Noise


            sampled_points.append(
                SpatioTemporalPoint(
                    lon=units.Distance(deg=coordinates[0]),
                    lat=units.Distance(deg=coordinates[1]),
                    date_time=datetime.datetime.fromtimestamp(
                        int(planner.current_data_t_0 + planner.reach_times[0]),
                        tz=datetime.timezone.utc,
                    ),
                )
            )
        return sampled_points

    def plot_batch(self, filename: str, random_sample_points: Optional[int] = 10):
        plot_start_time = time.time()

        last_target = self.problems[-1].end_region

        def add_drawing(ax: plt.axis, rel_time_in_seconds):
            # Arena Frame
            self.arena.plot_arena_frame_on_map(ax)
            # Target Frame
            ax.add_patch(
                patches.Rectangle(
                    (self.target_x_start, self.target_y_start),
                    (self.target_x_end - self.target_x_start),
                    (self.target_y_end - self.target_y_start),
                    linewidth=4,
                    edgecolor="g",
                    facecolor="none",
                    label="target sampling frame",
                )
            )
            # Starts Frame
            ax.add_patch(
                patches.Rectangle(
                    (self.starts_x_start, self.starts_y_start),
                    (self.starts_x_end - self.starts_x_start),
                    (self.starts_y_end - self.starts_y_start),
                    linewidth=4,
                    edgecolor="g",
                    facecolor="none",
                    label="start sampling frame",
                )
            )
            # Planner Frame
            ax.add_patch(
                patches.Rectangle(
                    (
                        self.hindcast_planner.grid.domain.lo[0],
                        self.hindcast_planner.grid.domain.lo[1],
                    ),
                    (
                        self.hindcast_planner.grid.domain.hi[0]
                        - self.hindcast_planner.grid.domain.lo[0]
                    ),
                    (
                        self.hindcast_planner.grid.domain.hi[1]
                        - self.hindcast_planner.grid.domain.lo[1]
                    ),
                    linewidth=4,
                    edgecolor="b",
                    facecolor="none",
                    label="hj solver frame",
                )
            )

            # Minimal Distance to Target
            ax.add_patch(
                plt.Circle(
                    (last_target.lon.deg, last_target.lat.deg),
                    self.config["target_min_distance"],
                    color="r",
                    linewidth=2,
                    facecolor="none",
                )
            )

            # Add Starts to Plot
            for problem in self.problems:
                ax.scatter(
                    problem.start_state.lon.deg,
                    problem.start_state.lat.deg,
                    facecolors="none",
                    edgecolors="r",
                    marker="o",
                    label="starts",
                )

            # Plot more possible Starts
            if random_sample_points:
                for point in self.generate_starts(amount=random_sample_points, silent=True):
                    ax.scatter(
                        point.lon.deg,
                        point.lat.deg,
                        facecolors="none",
                        edgecolors="black",
                        marker="o",
                        label="possible sample points",
                    )

            ax.set_title(f"Multi-Reach at time ({rel_time_in_seconds/3600:.1f}h)")

        self.hindcast_planner.plot_reachability_animation(
            filename=filename,
            add_drawing=add_drawing,
        )

        if self.verbose > 0:
            print(
                f"MissionGenerator: Batch of {len(self.problems)} plotted ({time.time()-plot_start_time:.1f}s)"
            )

    def run_forecast(self, batch_folder):
        """
        We have to trick the planner here to run exactly the forecasts we need:
            - we take any problem of the current batch (they share the same target, so the planner is the same)
            - we run until timeout (problem.is_done == -1)
            - we reset the platform x&y to start coordinates s.t. we never runs out of area
        """
        problem_factory = FileProblemFactory(csv_file=f"{batch_folder}problems.csv")
        problem = problem_factory.next_problem()

        arena = ArenaFactory.create(scenario_file=self.scenario_file, problem=problem)

        forecast_planner = HJReach2DPlanner(
            problem=problem,
            specific_settings=self.hindcast_planner.specific_settings
            | {"planner_path": batch_folder, "save_after_planning": True},
        )

        # Run Arena until Timeout (reset position to no leave arena)
        observation = arena.reset(problem.start_state)
        while problem.is_done(observation.platform_state) != -1:
            observation = arena.step(forecast_planner.get_action(observation))
            arena.platform.state.lon = problem.start_state.lon
            arena.platform.state.lat = problem.start_state.lat
