import datetime
import logging
import math
import os
import shutil
import time
import traceback
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.data_sources.DataSource import (
    SubsettingDataSourceException,
)
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.ArenaFactory import (
    ArenaFactory,
    CorruptedOceanFileException,
    MissingOceanFileException,
)
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    PlatformStateSet,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.reinforcement_learning.missions.CachedNavigationProblem import (
    CachedNavigationProblem,
)
from ocean_navigation_simulator.utils import cluster_utils, units
from ocean_navigation_simulator.utils.misc import bcolors, timing_dict
from ocean_navigation_simulator.utils.units import Distance

logger = logging.getLogger("MissionGenerator")


class MissionGenerator:
    """
    A flexible class to generate missions.
    Each MissionGenerator will try&error until feasible targets/starts are found
     - if parallelized each MissionGenerator should be feed with a different seed and cache_folder for plots
     - problem is uniquely identified by seed/factory index
    """

    def __init__(self, config: Optional[dict] = {}, c3=None):
        # update the config to be in proper units
        if type(config["t_range"][0]) == str:
            config["t_range"] = [datetime.datetime.fromisoformat(t) for t in config["t_range"]]
        self.config = config
        self.c3 = c3
        self.random = np.random.default_rng(self.config["seed"])

    def cache_batch(self) -> Tuple[List[CachedNavigationProblem], dict, List]:
        """Generate Batch, Cache Hindcast, Cache Forecast and Plot Batch"""
        self.performance = {
            "total_time": 0,
            "generate_time": 0,
            "plot_time": 0,
            "forecast_time": 0,
            "hindcast_time": 0,
            "starts": 0,
            "errors": 0,
            "target_resampling": 0,
            "start_resampling": 0,
        }
        self.errors = []

        with timing_dict(
            self.performance,
            "total_time",
            "Batch finished ({})",
            logger,
        ):
            while True:
                try:
                    if self.config["cache_folder"].endswith("/"):
                        os.makedirs(self.config["cache_folder"], exist_ok=True)

                    # Step 1: Generate Batch
                    with timing_dict(
                        self.performance,
                        "generate_time",
                        "Generated Batch ({})",
                        logger,
                    ):
                        if self.config["multi_agent"]["is_multi_agent"]:
                            problems = self.generate_batch_multi_agent()
                        else:
                            problems = self.generate_batch()

                    # Step 2: Plot or Animate Batch
                    if self.config["animate_batch"]:
                        with timing_dict(
                            self.performance,
                            "plot_time",
                            "Animated Batch ({})",
                            logger,
                        ):
                            if self.config["cache_folder"].startswith("/seaweed-storage/"):
                                cluster_utils.ensure_storage_connection()
                            self.plot_last_batch_animation(
                                filename=f"{self.config['cache_folder']}animation.gif"
                            )

                    if self.config["plot_batch"]:
                        with timing_dict(
                            self.performance,
                            "plot_time",
                            "Plotted Batch ({})",
                            logger,
                        ):
                            if self.config["cache_folder"].startswith("/seaweed-storage/"):
                                cluster_utils.ensure_storage_connection()
                            self.plot_last_batch_snapshot(
                                filename=f"{self.config['cache_folder']}snapshot.png"
                            )

                    # step 3: Cache Hindcast Planner
                    if self.config["cache_hindcast"]:
                        with timing_dict(
                            self.performance,
                            "hindcast_time",
                            "Cached Hindcast ({})",
                            logger,
                        ):
                            if self.config["cache_folder"].startswith("/seaweed-storage/"):
                                cluster_utils.ensure_storage_connection()
                            self.hindcast_planner.save_planner_state(
                                f"{self.config['cache_folder']}hindcast_planner/"
                            )

                    # # Step 4: Cache Forecast Planner
                    if self.config["cache_forecast"]:
                        with timing_dict(
                            self.performance,
                            "forecast_time",
                            "Cached Forecast ({})",
                            logger,
                        ):
                            self.run_forecast(problem=problems[0])
                except Exception as e:
                    if hasattr(self, "problems") and len(self.problems):
                        logger.warning(bcolors.red(self.problems[0]))
                    logger.warning(bcolors.red(str(e)))
                    logger.warning(traceback.format_exc())
                    self.performance["errors"] += 1
                    self.errors.append(str(e))
                    shutil.rmtree(self.config["cache_folder"], ignore_errors=True)
                else:
                    break

        self.performance["starts"] = len(self.problems)

        return self.problems, self.performance, self.errors

    def generate_batch(self) -> List[CachedNavigationProblem]:
        """Generate a target and a batch of starts according to config"""
        self.problems = []

        # Step 1: Generate Target & Starts until enough valid target/starts found
        while not (target := self.generate_target()) or not (
            starts := self.generate_starts()
        ):  # if self.config["multi_agent"] else starts := ):
            pass

        # Step 2: Create Problems
        for random, start, distance_to_shore in starts:
            ttr = self.hindcast_planner.interpolate_value_function_in_hours(point=start).item()
            feasible = ttr < self.config["problem_timeout_in_h"]

            self.problems.append(
                CachedNavigationProblem(
                    start_state=PlatformState(
                        lon=start.lon,
                        lat=start.lat,
                        date_time=start.date_time,
                    ),
                    end_region=target.to_spatial_point(),
                    target_radius=self.config["problem_target_radius"],
                    platform_dict=self.arena.platform.platform_dict,
                    extra_info={
                        "multi_agent": self.config["multi_agent"]["is_multi_agent"],
                        "timeout_datetime": target.date_time.isoformat(),
                        "start_target_distance_deg": target.haversine(start).deg,
                        "feasible": feasible,
                        "ttr_in_h": ttr,
                        "random": random,
                        "distance_to_shore_deg": distance_to_shore.deg,
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
            logger.debug(f"Problem created: {self.problems[-1]}")

        return self.problems

    def generate_batch_multi_agent(self) -> List[CachedNavigationProblem]:
        """Generate a target and a batch of starts according to config"""
        self.problems = []
        # Step 1: Generate Target & Starts until enough valid target/starts found
        while not (target := self.generate_target()) or not (
            starts := self.generate_multi_agent_starts()
        ):
            pass
        missions, start_points = starts[0], starts[1]
        for k in range(len(missions)):
            ttr_k = [
                float(self.hindcast_planner.interpolate_value_function_in_hours(point=start))
                for _, start, _ in missions[k]
            ]  # output of interpolate is a single element array, cast it to scalar
            feasible = max(ttr_k) < self.config["problem_timeout_in_h"]
            self.problems.append(
                CachedNavigationProblem(
                    start_state=PlatformStateSet(
                        platform_list=[
                            PlatformState(
                                lon=start.lon,
                                lat=start.lat,
                                date_time=start.date_time,
                            )
                            for _, start, _ in missions[k]
                        ]
                    ),
                    end_region=target.to_spatial_point(),
                    target_radius=self.config["problem_target_radius"],
                    platform_dict=self.arena.platform.platform_dict,
                    extra_info={
                        "multi_agent": self.config["multi_agent"]["is_multi_agent"],
                        "timeout_datetime": target.date_time.isoformat(),
                        "start_target_distance_deg": [
                            target.haversine(start).deg for _, start, _ in missions[k]
                        ],
                        "feasible": feasible,
                        "ttr_in_h": max(ttr_k),
                        "all_ttr_among_platforms_in_h": ttr_k,
                        "random": missions[k][0][
                            0
                        ],  # Don't need to pass a list here, random same for all multi-agent start points #[random for random,_,_ in missions[k]],
                        "distance_to_shore_deg": [
                            distance_to_shore.deg for _, _, distance_to_shore in missions[k]
                        ],
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
                deg=self.random.uniform(self.config["x_range"][0], self.config["x_range"][1])
            ),
            lat=units.Distance(
                deg=self.random.uniform(self.config["y_range"][0], self.config["y_range"][1])
            ),
            # only sample goal times s.t. all missions will start and timeout in t_interval
            date_time=datetime.datetime.fromtimestamp(
                self.random.uniform(
                    (
                        self.config["t_range"][0]
                        + datetime.timedelta(hours=self.config["problem_timeout_in_h"])
                    ).timestamp(),
                    self.config["t_range"][1].timestamp(),
                ),
                tz=datetime.timezone.utc,
            ),
        )
        fake_start = PlatformState(
            lon=fake_target.lon,
            lat=fake_target.lat,
            date_time=fake_target.date_time
            - datetime.timedelta(hours=self.config["problem_timeout_in_h"]),
        )

        ##### Step 2: Reject if files are missing or corrupted #####
        try:
            self.arena = ArenaFactory.create(
                scenario_file=self.config.get("scenario_file", None),
                scenario_config=self.config.get("scenario_config", None),
                x_interval=[units.Distance(deg=x) for x in self.config["x_range"]],
                y_interval=[units.Distance(deg=y) for y in self.config["y_range"]],
                t_interval=[
                    fake_start.date_time - datetime.timedelta(hours=1),
                    fake_target.date_time + datetime.timedelta(days=1, hours=1),
                ],
                throw_exceptions=True,
                c3=self.c3,
            )
            self.arena.reset(platform_set=PlatformStateSet(platform_list=[fake_start]))
        except (MissingOceanFileException, CorruptedOceanFileException) as e:
            logger.warning(
                f"Target aborted because of missing or corrupted files: [{fake_start.date_time}, {fake_target.date_time}]."
            )
            logger.warning(e)
            self.performance["errors"] += 1
            self.errors.append(str(e))
            return False

        # Step 3: Reject if to close to land
        distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
            fake_target.to_spatial_point()
        )
        if distance_to_shore.deg < self.config["target_distance_from_land"]:
            logger.warning(
                f"Target aborted because too close to land: {fake_target.to_spatial_point()} = {distance_to_shore}."
            )
            self.performance["target_resampling"] += 1
            return False

        ##### Step 3: Run multi-time-back HJ Planner #####
        try:
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
            with timing_dict(
                self.performance,
                "hindcast_time",
                "Run Hindcast Planner ({})",
                logger,
                logging.INFO,
            ):
                self.hindcast_planner = HJReach2DPlanner(
                    problem=NavigationProblem(
                        start_state=fake_start,
                        end_region=fake_target.to_spatial_point(),
                        target_radius=self.config["problem_target_radius"],
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

        except SubsettingDataSourceException as e:
            logger.warning(
                "Target aborted because of subsetting exception: [{fake_start.date_time}, {fake_target.date_time}]."
            )
            logger.warning(e)
            self.performance["errors"] += 1
            self.errors.append(str(e))
            return False

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
        logger.info(f"Target created: {real_target} ({time.time()-start_time:.1f}s).")

        return real_target

    def get_sampling_frame_mission_time(self) -> Tuple[dict, list]:
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
            int(self.config["feasible_mission_time_in_h"][0] * 3600),
            int(self.config["feasible_mission_time_in_h"][1] * 3600),
        ]
        return sampling_frame, mission_time

    def generate_multi_agent_starts(self) -> List[List[Tuple[bool, SpatioTemporalPoint, float]]]:
        start_time = time.time()
        sampling_frame, mission_time = self.get_sampling_frame_mission_time()
        feasible_missions, feasible_points = self.sample_connected_feasible_points(
            sampling_frame=sampling_frame, mission_time=mission_time
        )
        if len(feasible_points) < self.config["feasible_missions_per_target"]:
            logger.warning(
                f"Only {len(feasible_points)}/{self.config['feasible_missions_per_target']} feasible missions available."
            )
        return feasible_missions, feasible_points

    def generate_starts(self) -> Union[List[Tuple[bool, SpatioTemporalPoint, float]], bool]:
        """
        Samples in/feasible starts from the already generated target.
        Returns:
            List of starts for the pre-existing target.
        """
        start_time = time.time()
        sampling_frame, mission_time = self.get_sampling_frame_mission_time()
        feasible_points = self.sample_feasible_points(sampling_frame, mission_time)
        random_points = self.sample_random_points(sampling_frame)

        if len(feasible_points) < self.config["feasible_missions_per_target"]:
            logger.warning(
                f"Only {len(feasible_points)}/{self.config['feasible_missions_per_target']} feasible points available."
            )
        elif len(random_points) < self.config["random_missions_per_target"]:
            logger.warning(
                f"Only {len(random_points)}/{self.config['random_missions_per_target']} random points available."
            )
        else:
            logger.info(
                f"{len(feasible_points + random_points)} starts created ({time.time()-start_time:.1f}s)."
            )

        return feasible_points + random_points

    def sample_connected_feasible_points(
        self, sampling_frame, mission_time
    ) -> List[Tuple[bool, SpatioTemporalPoint, Distance]]:

        planner = self.hindcast_planner
        feasible_missions = []
        list_spatio_temp_points = []
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

        logger.info(
            "Sampling Ratio: {n}/{d} = {ratio:.2%}".format(
                n=points_to_sample.shape[0],
                d=planner.all_values[0].size,
                ratio=points_to_sample.shape[0] / planner.all_values[0].size,
            )
        )
        sampling_radius = units.Distance(km=self.config["multi_agent"]["sampling_range_radius_km"])
        nb_mission_found = 0
        # Step 2: Return List of SpatioTemporalPoint
        while nb_mission_found < min(
            self.config["feasible_missions_per_target"], points_to_sample.shape[0]
        ):
            self.performance["start_resampling"] = 0
            sample_index = self.random.integers(points_to_sample.shape[0])
            sampled_point = points_to_sample[sample_index]
            points_to_sample = np.delete(points_to_sample, sample_index, axis=0)
            coordinates = planner.grid.states[sampled_point[0], sampled_point[1], :]
            mission_connected_points = []
            while len(mission_connected_points) < self.config["multi_agent"]["nb_platforms"]:
                # Add Noise to current point
                angle_noise = self.random.uniform(0, 2 * np.pi)
                radius_noise = self.random.uniform(0, sampling_radius.deg)
                # Format
                point = SpatioTemporalPoint(
                    lon=units.Distance(deg=coordinates[0] + radius_noise * np.cos(angle_noise)),
                    lat=units.Distance(deg=coordinates[1] + radius_noise * np.sin(angle_noise)),
                    # Smallest available time in hj planner
                    date_time=datetime.datetime.fromtimestamp(
                        math.ceil(planner.current_data_t_0 + planner.reach_times[0]),
                        tz=datetime.timezone.utc,
                    ),
                )
                point_ttr_h = float(
                    self.hindcast_planner.interpolate_value_function_in_hours(point=point)
                )
                # Add if far enough from shore
                distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
                    point.to_spatial_point()
                )

                if (
                    (distance_to_shore.deg > self.config["min_distance_from_land"])
                    & (self.config["multi_agent"]["sample_range_ttr_h"][0] < point_ttr_h)
                    & (point_ttr_h < (self.config["multi_agent"]["sample_range_ttr_h"][1]))
                ):

                    if len(mission_connected_points) == 0:
                        mission_connected_points.append((False, point, distance_to_shore))
                        sampled_spatial_points = np.array(point.to_spatial_point()).reshape(2, -1)
                    else:
                        dist_to_platf = point.to_spatial_point().haversine(
                            SpatialPoint(
                                lon=units.Distance(deg=sampled_spatial_points[0]),
                                lat=units.Distance(deg=sampled_spatial_points[1]),
                            )
                        )
                        if any(  # connected to at least one of the other members
                            self.config["scenario_config"]["multi_agent_constraints"][
                                "collision_thrsld"
                            ]
                            < dist.km
                            < self.config["scenario_config"]["multi_agent_constraints"][
                                "communication_thrsld"
                            ]
                            for dist in dist_to_platf
                        ):
                            mission_connected_points.append((False, point, distance_to_shore))
                            sampled_spatial_points = np.append(
                                sampled_spatial_points,
                                np.array(point.to_spatial_point()).reshape(2, -1),
                                axis=1,
                            )
                else:
                    self.performance["start_resampling"] += 1
                    if (
                        self.performance["start_resampling"] > 20
                    ):  # if actual point not a good one, change
                        break
            if (
                len(mission_connected_points) == self.config["multi_agent"]["nb_platforms"]
            ):  # only append if we were able to find enough starts for multi-agents
                feasible_missions.append(mission_connected_points)
                list_spatio_temp_points.append(
                    SpatioTemporalPoint(
                        lon=units.Distance(deg=sampled_spatial_points[0]),
                        lat=units.Distance(deg=sampled_spatial_points[1]),
                        date_time=np.repeat(
                            datetime.datetime.fromtimestamp(
                                math.ceil(planner.current_data_t_0 + planner.reach_times[0]),
                                tz=datetime.timezone.utc,
                            ),
                            self.config["multi_agent"]["nb_platforms"],
                        ),
                    )
                )
                nb_mission_found += 1
            else:
                continue
        print("feasible missions found")
        return feasible_missions, list_spatio_temp_points

    def sample_feasible_points(
        self, sampling_frame, mission_time
    ) -> List[Tuple[bool, SpatioTemporalPoint, Distance]]:
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

        logger.info(
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
                    math.ceil(planner.current_data_t_0 + planner.reach_times[0]),
                    tz=datetime.timezone.utc,
                ),
            )

            # Add if far enough from shore
            distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
                point.to_spatial_point()
            )
            if distance_to_shore.deg > self.config["min_distance_from_land"]:
                sampled_points.append((False, point, distance_to_shore))
            else:
                self.performance["start_resampling"] += 1
            if len(sampled_points) >= self.config["feasible_missions_per_target"]:
                break

        return sampled_points

    def sample_random_points(
        self, sampling_frame
    ) -> List[Tuple[bool, SpatioTemporalPoint, Distance]]:
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
                sampled_points.append((True, point, distance_to_shore))
            else:
                self.performance["start_resampling"] += 1
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax = self.hindcast_planner.plot_reachability_snapshot_over_currents(return_ax=True)
        self.add_problems_to_ax(ax)

        if filename is None:
            ax.get_figure().show()
        else:
            ax.get_figure().savefig(filename)

    def plot_last_batch_animation(self, filename):
        if len(self.problems) == 0:
            raise Exception("plot_last_batch_animation can only be called after generate_batch.")

        def add_Drawing(*args):
            self.add_problems_to_ax(*args)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.hindcast_planner.plot_reachability_animation(
                filename=filename,
                add_drawing=add_Drawing,
            )

    def run_forecast(self, problem):
        """
        We have to trick the planner here to run exactly the forecasts we need:
            - we take any problem of the current batch (they share the same target, so the planner is the same)
            - we run until timeout (problem.is_done == -1)
            - we reset the platform x&y to start coordinates s.t. we never runs out of area
        """
        forecast_planner = HJReach2DPlanner(
            problem=problem,
            specific_settings=self.hindcast_planner.specific_settings
            | self.hj_planner_frame
            | {"planner_path": self.config["cache_folder"], "save_after_planning": True},
        )

        # # Run Arena until Timeout (use start position to not leave arena)
        # sim_date_time = problem.start_state.date_time
        # while sim_date_time < problem.start_state.date_time + problem.timeout:
        #     forecast_planner.replan_if_necessary(
        #         ArenaObservation(
        #             platform_state=PlatformState(
        #                 lon=problem.start_state.lon,
        #                 lat=problem.start_state.lat,
        #                 date_time=sim_date_time,
        #             ),
        #             true_current_at_state=self.arena.ocean_field.get_ground_truth(
        #                 problem.start_state.to_spatio_temporal_point()
        #             ),
        #             forecast_data_source=self.arena.ocean_field.forecast_data_source,
        #         )
        #     )
        #     sim_date_time += datetime.timedelta(minutes=10)

        arena = ArenaFactory.create(
            scenario_file=self.config["scenario_file"],
            problem=problem,
            throw_exceptions=True,
            c3=self.c3,
        )

        # Run Arena until Timeout (reset position to no leave arena)
        observation = arena.reset(problem.start_state)

        while problem.is_done(observation.platform_state) != -1:
            observation = arena.step(forecast_planner.get_action(observation))
            arena.platform.state.lon = problem.start_state.lon
            arena.platform.state.lat = problem.start_state.lat
