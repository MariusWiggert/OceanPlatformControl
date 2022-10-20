import datetime
import math
import time
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatioTemporalPoint
from ocean_navigation_simulator.reinforcement_learning.missions.CachedNavigationProblem import (
    CachedNavigationProblem,
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

    def generate_batch(self) -> [CachedNavigationProblem]:
        """Generate a target and a batch of starts according to config"""
        self.problems = []

        with timing("MissionGenerator: Batch created ({:.1f}s)", self.verbose):
            # Step 1: Generate Target & Starts until enough valid target/starts found
            while not (target := self.generate_target()) or not (starts := self.generate_starts()):
                pass

            # Step 2: Create Problems
            for random, start in starts:
                ttr = self.hindcast_planner.interpolate_value_function_in_hours(point=start).item()
                feasible = ttr < self.config["problem_timeout"].total_seconds() / 3600

                self.problems.append(
                    CachedNavigationProblem(
                        start_state=PlatformState(
                            lon=start.lon,
                            lat=start.lat,
                            date_time=start.date_time,
                        ),
                        end_region=target.to_spatial_point(),
                        target_radius=self.config["problem_target_radius"],
                        timeout=(target.date_time - start.date_time),
                        platform_dict=self.arena.platform.platform_dict,
                        extra_info={
                            "timeout_datetime": target.date_time.isoformat(),
                            "start_target_distance_deg": target.haversine(start).deg,
                            "feasible": feasible,
                            "ttr_in_h": ttr,
                            "random": random,
                            # Planner Caching
                            "x_cache": self.hj_planner_frame["x_interval"],
                            "y_cache": self.hj_planner_frame["y_interval"],
                            # "t_cache": self.hj_planner_time_frame,
                            # Factory
                            "factory_seed": self.config["seed"],
                            "factory_index": len(self.problems),
                        },
                    )
                )
                if self.verbose > 2:
                    print(f"MissionGenerator: Problem created: {self.problems[-1]}")

        return self.problems

    def generate_target(self) -> Union[SpatioTemporalPoint, bool]:
        """
        Samples a target in x_range/y_range/t_range.
        - ensures that all required files available on c3
        - ensures distance to shore
        - runs hj planner for target

        Returns:
            SpatioTemporalPoint to be a target
        """
        start_time = time.time()

        ##### Step 1: Generate Target (x,y,t_T) #####
        # Planner starts from timeout backwards (this is a trick, so we can use the planner after max_mission_range)!
        fake_target = SpatioTemporalPoint(
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
                    (self.config["t_range"][0] + self.config["problem_timeout"]).timestamp(),
                    self.config["t_range"][1].timestamp(),
                ),
                tz=datetime.timezone.utc,
            ),
        )
        fake_start = PlatformState(
            lon=fake_target.lon,
            lat=fake_target.lat,
            date_time=fake_target.date_time - self.config["problem_timeout"],
        )

        ##### Step 2: Reject if files are missing or corrupted #####
        try:
            self.arena = ArenaFactory.create(
                scenario_file=self.scenario_file,
                scenario_config={
                    'ocean_dict': {
                        'hindcast': {

                        }
                    }
                },
                x_interval=self.config["x_range"],
                y_interval=self.config["y_range"],
                t_interval=[
                    fake_start.date_time - datetime.timedelta(days=1),
                    fake_target.date_time + datetime.timedelta(days=1),
                ],
                verbose=self.verbose - 2,
            )
        except (MissingOceanFileException, CorruptedOceanFileException):
            if self.verbose > 1:
                print(
                    bcolors.red(
                        f"MissionGenerator: Target aborted because of missing or corrupted files: [{fake_start.date_time}, {fake_target.date_time}]."
                    )
                )
            return False

        # Step 3: Reject if to close to land
        distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
            fake_target.to_spatial_point()
        )
        if distance_to_shore.deg < self.config["target_distance_from_land"]:
            if self.verbose > 1:
                print(
                    bcolors.red(
                        f"MissionGenerator: Target aborted because too close to land: {fake_target.to_spatial_point()}."
                    )
                )
            return False

        ##### Step 3: Run multi-time-back HJ Planner #####
        self.hj_planner_frame = {
            "x_interval": [
                fake_target.lon.deg - self.config["hj_planner_box"],
                fake_target.lon.deg + self.config["hj_planner_box"],
            ],
            "y_interval": [
                fake_target.lat.deg - self.config["hj_planner_box"],
                fake_target.lat.deg + self.config["hj_planner_box"],
            ],
        }
        self.hindcast_planner = HJReach2DPlanner(
            problem=NavigationProblem(
                start_state=fake_start,
                end_region=fake_target.to_spatial_point(),
                target_radius=self.config["problem_target_radius"],
                timeout=self.config["problem_timeout"],
                platform_dict=self.arena.platform.platform_dict,
            ),
            specific_settings=self.config["hj_specific_settings"] | self.hj_planner_frame,
        )
        self.hindcast_planner.replan_if_necessary(
            ArenaObservation(
                platform_state=fake_start,
                true_current_at_state=self.arena.ocean_field.get_ground_truth(
                    fake_start.to_spatio_temporal_point()
                ),
                forecast_data_source=self.arena.ocean_field.hindcast_data_source,
            )
        )

        ##### Overwrite with biggest available time in hj planner #####
        real_target = SpatioTemporalPoint(
            lon=fake_start.lon,
            lat=fake_start.lat,
            # Largest available time in hj planner
            date_time=datetime.datetime.fromtimestamp(
                math.floor(
                    self.hindcast_planner.current_data_t_0 + self.hindcast_planner.reach_times[-1]
                ),
                tz=datetime.timezone.utc,
            ),
        )

        if self.verbose > 1:
            print(
                bcolors.green(
                    f"MissionGenerator: Target created: {real_target} ({time.time()-start_time:.1f}s)."
                )
            )

        return real_target

    def generate_starts(self) -> Union[List[Tuple[bool, SpatioTemporalPoint]], bool]:
        """
        Samples in/feasible starts from the already generated target.
        Returns:
            List of starts for the pre-existing target.
        """
        start_time = time.time()

        sampling_frame = {
            "x": [
                self.hj_planner_frame["x_interval"][0] + self.config["min_distance_from_hj_frame"],
                self.hj_planner_frame["x_interval"][1] - self.config["min_distance_from_hj_frame"],
            ],
            "y": [
                self.hj_planner_frame["y_interval"][0] + self.config["min_distance_from_hj_frame"],
                self.hj_planner_frame["y_interval"][1] - self.config["min_distance_from_hj_frame"],
            ],
        }
        mission_time = [
            int(self.config["feasible_mission_time"][0].total_seconds()),
            int(self.config["feasible_mission_time"][1].total_seconds()),
        ]
        feasible_points = self.sample_feasible_points(sampling_frame, mission_time)
        random_points = self.sample_random_points(sampling_frame)

        if self.verbose > 1:
            if len(feasible_points) < self.config["feasible_missions_per_target"]:
                print(
                    bcolors.red(
                        f"MissionGenerator: Only {len(feasible_points)}/{self.config['feasible_missions_per_target']} feasible points available."
                    )
                )
            elif len(random_points) < self.config["random_missions_per_target"]:
                print(
                    bcolors.red(
                        f"MissionGenerator: Only {len(random_points)}/{self.config['random_missions_per_target']} random points available."
                    )
                )
            else:
                print(
                    bcolors.green(
                        f"MissionGenerator: {len(feasible_points + random_points)} starts created ({time.time()-start_time:.1f}s)."
                    )
                )

        return feasible_points + random_points

    def sample_feasible_points(
        self, sampling_frame, mission_time
    ) -> List[Tuple[bool, SpatioTemporalPoint]]:
        planner = self.hindcast_planner

        # Step 1: Find reachable points with minimum distance from frame
        reach_times = (planner.all_values[0] - planner.all_values.min()) * (
            planner.current_data_t_T - planner.current_data_t_0
        )
        reachable_condition = (mission_time[0] < reach_times) & (reach_times < mission_time[1])
        frame_condition_x = (sampling_frame["x"][0] < planner.grid.states[:, :, 0]) & (
            planner.grid.states[:, :, 0] < sampling_frame["x"][1]
        )
        frame_condition_y = (sampling_frame["y"][0] < planner.grid.states[:, :, 1]) & (
            planner.grid.states[:, :, 1] < sampling_frame["y"][1]
        )
        points_to_sample = np.argwhere(reachable_condition & frame_condition_x & frame_condition_y)

        if self.verbose > 1:
            print(
                "Sampling Ratio: {n}/{d} = {ratio:.2%}".format(
                    n=points_to_sample.shape[0],
                    d=planner.all_values[0].size,
                    ratio=points_to_sample.shape[0] / planner.all_values[0].size,
                )
            )

        # Step 2: Return List of SpatioTemporalPoint
        sampled_points = []
        for _ in range(
            min(5 * self.config["feasible_missions_per_target"], points_to_sample.shape[0])
        ):
            # Sample Coordinates
            sample_index = self.random.integers(points_to_sample.shape[0])
            sampled_point = points_to_sample[sample_index]
            points_to_sample = np.delete(points_to_sample, sample_index, axis=0)
            coordinates = planner.grid.states[sampled_point[0], sampled_point[1], :]

            # Add Noise
            noise = (
                self.config["hj_specific_settings"]["grid_res"] * self.random.uniform(-0.5, 0.5),
                self.config["hj_specific_settings"]["grid_res"] * self.random.uniform(-0.5, 0.5),
            )

            # Format
            point = SpatioTemporalPoint(
                lon=units.Distance(deg=coordinates[0] + noise[0]),
                lat=units.Distance(deg=coordinates[1] + noise[1]),
                # Smallest available time in hj planner
                date_time=datetime.datetime.fromtimestamp(
                    math.ceil(
                        self.hindcast_planner.current_data_t_0
                        + self.hindcast_planner.reach_times[0]
                    ),
                    tz=datetime.timezone.utc,
                ),
            )

            # Add if far enough from shore
            distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
                point.to_spatial_point()
            )
            if distance_to_shore.deg > self.config["min_distance_from_land"]:
                sampled_points.append((False, point))
            if len(sampled_points) >= self.config["feasible_missions_per_target"]:
                break

        return sampled_points

    def sample_random_points(self, sampling_frame) -> List[Tuple[bool, SpatioTemporalPoint]]:
        """uniform Sampling in Frame"""
        sampled_points = []

        for _ in range(2 * self.config["random_missions_per_target"]):
            point = SpatioTemporalPoint(
                lon=units.Distance(
                    deg=self.random.uniform(sampling_frame["x"][0], sampling_frame["x"][1])
                ),
                lat=units.Distance(
                    deg=self.random.uniform(sampling_frame["y"][0], sampling_frame["y"][1])
                ),
                # Smallest available time in hj planner
                date_time=datetime.datetime.fromtimestamp(
                    math.ceil(
                        self.hindcast_planner.current_data_t_0
                        + self.hindcast_planner.reach_times[0]
                    ),
                    tz=datetime.timezone.utc,
                ),
            )

            # Add if far enough from shore
            distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
                point.to_spatial_point()
            )
            if distance_to_shore.deg > self.config["min_distance_from_land"]:
                sampled_points.append((True, point))
            if len(sampled_points) >= self.config["random_missions_per_target"]:
                break

        return sampled_points

    def add_problems_to_ax(self, ax: plt.axis, rel_time_in_seconds=0):
        # Add Starts to Plot
        for problem in self.problems:
            ax.scatter(
                problem.start_state.lon.deg,
                problem.start_state.lat.deg,
                facecolors="black" if problem.extra_info["random"] else "none",
                edgecolors="black" if problem.extra_info["random"] else "r",
                marker="o",
                label="starts",
            )

    def plot_last_batch_snapshot(self, filename: str = None):
        if len(self.problems) == 0:
            raise Exception("plot_last_batch_snapshot can only be called after generate_batch.")

        ax = self.hindcast_planner.plot_reachability_snapshot_over_currents(return_ax=True)
        self.add_problems_to_ax(ax)
        ax.get_figure().show()

    def plot_last_batch_animation(self, filename):
        with timing("MissionGenerator: Batch plotted ({:.1f}s)", self.verbose):
            if len(self.problems) == 0:
                raise Exception(
                    "plot_last_batch_animation can only be called after generate_batch."
                )

            def add_Drawing(*args):
                self.add_problems_to_ax(*args)

            self.hindcast_planner.plot_reachability_animation(
                filename=filename,
                add_drawing=add_Drawing,
            )

    #
    # def plot_batch_animation(self, filename: str, random_sample_points: Optional[int] = 10):
    #     plot_start_time = time.time()
    #
    #     last_target = self.problems[-1].end_region
    #
    #     def add_drawing():
    #         # Arena Frame
    #         self.arena.plot_arena_frame_on_map(ax)
    #         # Target Frame
    #         # ax.add_patch(
    #         #     patches.Rectangle(
    #         #         (self.target_x_start, self.target_y_start),
    #         #         (self.target_x_end - self.target_x_start),
    #         #         (self.target_y_end - self.target_y_start),
    #         #         linewidth=4,
    #         #         edgecolor="g",
    #         #         facecolor="none",
    #         #         label="target sampling frame",
    #         #     )
    #         # )
    #         # Starts Frame
    #         # ax.add_patch(
    #         #     patches.Rectangle(
    #         #         (self.starts_x_start, self.starts_y_start),
    #         #         (self.starts_x_end - self.starts_x_start),
    #         #         (self.starts_y_end - self.starts_y_start),
    #         #         linewidth=4,
    #         #         edgecolor="g",
    #         #         facecolor="none",
    #         #         label="start sampling frame",
    #         #     )
    #         # )
    #         # Planner Frame
    #         ax.add_patch(
    #             patches.Rectangle(
    #                 (
    #                     self.hindcast_planner.grid.domain.lo[0],
    #                     self.hindcast_planner.grid.domain.lo[1],
    #                 ),
    #                 (
    #                     self.hindcast_planner.grid.domain.hi[0]
    #                     - self.hindcast_planner.grid.domain.lo[0]
    #                 ),
    #                 (
    #                     self.hindcast_planner.grid.domain.hi[1]
    #                     - self.hindcast_planner.grid.domain.lo[1]
    #                 ),
    #                 linewidth=4,
    #                 edgecolor="b",
    #                 facecolor="none",
    #                 label="hj solver frame",
    #             )
    #         )
    #
    #         # Minimal Distance to Target
    #         ax.add_patch(
    #             plt.Circle(
    #                 (last_target.lon.deg, last_target.lat.deg),
    #                 self.config["target_min_distance"],
    #                 color="r",
    #                 linewidth=2,
    #                 facecolor="none",
    #             )
    #         )
    #
    #         # Add Starts to Plot
    #         for problem in self.problems:
    #             ax.scatter(
    #                 problem.start_state.lon.deg,
    #                 problem.start_state.lat.deg,
    #                 facecolors="none",
    #                 edgecolors="r",
    #                 marker="o",
    #                 label="starts",
    #             )
    #
    #         # Plot more possible Starts
    #         # if random_sample_points:
    #         #     for point in self.generate_starts(amount=random_sample_points, silent=True):
    #         #         ax.scatter(
    #         #             point.lon.deg,
    #         #             point.lat.deg,
    #         #             facecolors="none",
    #         #             edgecolors="black",
    #         #             marker="o",
    #         #             label="possible sample points",
    #         #         )
    #
    #         ax.set_title(f"Multi-Reach at time ({rel_time_in_seconds/3600:.1f}h)")
    #
    #     self.hindcast_planner.plot_reachability_animation(
    #         filename=filename,
    #         add_drawing=add_drawing,
    #     )
    #
    #     if self.verbose > 0:
    #         print(
    #             f"MissionGenerator: Batch of {len(self.problems)} plotted ({time.time()-plot_start_time:.1f}s)"
    #         )

    def run_forecast(self, batch_folder, problem):
        """
        We have to trick the planner here to run exactly the forecasts we need:
            - we take any problem of the current batch (they share the same target, so the planner is the same)
            - we run until timeout (problem.is_done == -1)
            - we reset the platform x&y to start coordinates s.t. we never runs out of area
        """
        arena = ArenaFactory.create(scenario_file=self.scenario_file, problem=problem)

        forecast_planner = HJReach2DPlanner(
            problem=problem,
            specific_settings=self.hindcast_planner.specific_settings
            | self.hj_planner_frame
            | {"planner_path": batch_folder, "save_after_planning": True},
        )

        # Run Arena until Timeout (reset position to no leave arena)
        observation = arena.reset(problem.start_state)
        while problem.is_done(observation.platform_state) != -1:
            observation = arena.step(forecast_planner.get_action(observation))
            arena.platform.state.lon = problem.start_state.lon
            arena.platform.state.lat = problem.start_state.lat
