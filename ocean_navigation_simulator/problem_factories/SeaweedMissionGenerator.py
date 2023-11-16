import datetime
import logging
import os
import random
import time
import warnings
from typing import List, Optional, Tuple
from collections import deque


import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import haversine_distances

from ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner import (
    HJBSeaweed2DPlanner,
)

from ocean_navigation_simulator.environment.ArenaFactory import (
    ArenaFactory,
    CorruptedOceanFileException,
    MissingOceanFileException,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.environment.SeaweedProblem import (
    SeaweedProblem,
)
from ocean_navigation_simulator.utils import cluster_utils, units
from ocean_navigation_simulator.utils.misc import timing_dict


class SeaweedMissionGenerator:
    """
    The generator samples starting points (n_samples) within a given spatio-temporal region under some constraints (i.e. certain distance to land) and a given time horizon.
    """

    # TODO: test
    def __init__(self, config: dict = {}, c3=None):
        """Takes the config file, sets the seed and initializes the arena object.

        Args:
            config: dictionary which specifies the generation configuration
        """
        if type(config["t_range"][0]) == str:
            config["t_range"] = [datetime.datetime.fromisoformat(t) for t in config["t_range"]]
        self.config = config
        self.c3 = c3

        self.random = np.random.default_rng(self.config["seed"])

        self.logger = logging.getLogger("SeaweedMissionGenerator")

        self._initialize_arena_download_data()

        if not hasattr(self, "arena"):
            raise AttributeError("Arena is not implemented")

    def _initialize_arena_download_data(self):
        """Initializes the arena and downloads the data. Catches errors if download fails i.e. due to missing or corrupt files."""

        self.performance = {
            "total_time": 0,
            "generate_time": 0,
            "errors": 0,
        }

        self.errors = []
        try:
            self.arena = ArenaFactory.create(
                scenario_file=self.config.get("scenario_file", None),
                scenario_config=self.config.get("scenario_config", {}),
                scenario_name=self.config.get("scenario_name", None),
                x_interval=[units.Distance(deg=x) for x in self.config["x_range"]],
                y_interval=[units.Distance(deg=y) for y in self.config["y_range"]],
                t_interval=self.config["t_range"],
                throw_exceptions=True,
                c3=self.c3,
            )
        except (MissingOceanFileException, CorruptedOceanFileException) as e:
            self.logger.warning(
                f"Aborted because of missing or corrupted files: [{self.config['t_range'][0]}, {self.config['t_range'][1]}]."
            )
            self.logger.warning(e)
            self.performance["errors"] += 1
            self.errors.append(str(e))
            return False

    def generate_batch(self) -> List[SeaweedProblem]:
        """Generate a batch of starts according to config"""
        self.problems = []

        with timing_dict(
            self.performance,
            "total_time",
            "Batch finished ({})",
            self.logger,
        ):
            # Step 1: Generate starts
            starts = self._generate_starts()

            # Step 2: Create Problems
            for rand, start, distance_to_shore in starts:
                self.problems.append(
                    SeaweedProblem(
                        start_state=PlatformState(
                            lon=start.lon,
                            lat=start.lat,
                            date_time=start.date_time,
                        ),
                        platform_dict=self.arena.platform.platform_dict,
                        extra_info={
                            "time_horizon_in_sec": self.config["time_horizon"],
                            "random": rand,
                            "distance_to_shore_deg": distance_to_shore.deg,
                            # Factory
                            "factory_seed": self.config["seed"],
                            "factory_index": len(self.problems),
                        },
                    )
                )

        return self.problems

    def _generate_starts(self) -> List[Tuple[bool, SpatioTemporalPoint, float]]:
        """
        Samples in/feasible starts from the already generated target.
        Returns:
            List of starts for the pre-existing target.
        """
        start_time = time.time()
        sampled_points = []
        i = 0
        while i < self.config["n_samples"]:
            point = SpatioTemporalPoint(
                lon=units.Distance(
                    deg=self.random.uniform(self.config["x_range"][0], self.config["x_range"][1])
                ),
                lat=units.Distance(
                    deg=self.random.uniform(self.config["y_range"][0], self.config["y_range"][1])
                ),
                date_time=self._get_random_starting_time(),
            )
            # Add if far enough from shore
            distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
                point.to_spatial_point()
            )

            if distance_to_shore.deg > self.config[
                "min_distance_from_land"
            ] and self._point_in_pacific(point):
                sampled_points.append((True, point, distance_to_shore))
                i += 1

        self.logger.info(
            f"SeaweedMissionGenerator: {len(sampled_points)} starts created ({time.time()-start_time:.1f}s)."
        )

        return sampled_points

    def _get_random_starting_time(self):
        """Samples a random datetime object within the given time interval and ensures it is at least time_horizon earlier than the end of our time interval
        Returns:
            random_datetime
        """
        start_datetime, end_datetime = self.config["t_range"]
        end_datetime -= datetime.timedelta(seconds=self.config["time_horizon"])

        if end_datetime <= start_datetime:
            raise ValueError(
                "Invalid time interval, the time horizon is too large for the given time range"
            )

        random_seconds = random.uniform(0, (end_datetime - start_datetime).total_seconds())
        random_datetime = start_datetime + datetime.timedelta(seconds=random_seconds)

        return random_datetime

    @staticmethod
    def plot_starts(
        config: dict,
        problems: Optional[List] = None,
        results_folder: Optional[str] = None,
    ):

        # Step 1: Load Problems
        if problems is not None:
            problems_df = pd.DataFrame([problem.to_dict() for problem in problems])
        elif results_folder is not None:
            if results_folder.startswith("/seaweed-storage/"):
                cluster_utils.ensure_storage_connection()
            problems_df = pd.read_csv(f"{results_folder}problems.csv")
            analysis_folder = f"{results_folder}analysis/"
            os.makedirs(analysis_folder, exist_ok=True)
        else:
            raise ValueError(
                "Please provide a List of problems or a directory where the problems are saved in a .csv file as parameters."
            )
        problem = SeaweedProblem.from_pandas_row(problems_df.iloc[0])

        # Step 2:
        arena = ArenaFactory.create(
            scenario_file=config.get("scenario_file", None),
            scenario_config=config.get("scenario_config", {}),
            scenario_name=config.get("scenario_name", None),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
                x_interval=config["x_range"],
                y_interval=config["y_range"],
                time=problem.start_state.date_time,
                spatial_resolution=0.2,
                return_ax=True,
                # figsize=(32, 24),
            )
        ax.scatter(
            problems_df["x_0_lon"], problems_df["x_0_lat"], c="red", marker="o", s=6, label="starts"
        )

        ax.legend()
        if results_folder is not None:
            ax.get_figure().savefig(f"{analysis_folder}starts.png")
        ax.get_figure().show()

    def _point_in_pacific(self, point: SpatioTemporalPoint) -> bool:
        """Returns True if point in Region 3 is in Pacific, if in carribean sea returns False"""
        return not (
            (-90 < point.lon.deg < -70 and 15 < point.lat.deg < 20)
            or (-100 < point.lon.deg < -90 and 17.5 < point.lat.deg < 20)
            or (-85 < point.lon.deg < -70 and 10 < point.lat.deg < 15)
            or (-86 < point.lon.deg < -70 and 8.5 < point.lat.deg < 10)
        )


class SeaweedMissionGeneratorFileCheck:
    """
    Same as SeaweedMissionGenerator but including a check if all files are in the database.
    """
    def __init__(self, config: dict = {}, c3=None):
        """Takes the config file, sets the seed and initializes the arena object.

        Args:
            config: dictionary which specifies the generation configuration
        """
        if type(config["t_range"][0]) == str:
            config["t_range"] = [datetime.datetime.fromisoformat(t) for t in config["t_range"]]
        self.config = config
        self.c3 = c3

        self.random = np.random.default_rng(self.config["seed"])
        self.logger = logging.getLogger("SeaweedMissionGenerator")
        self.performance = {
            "total_time": 0,
            "generate_time": 0,
            "errors": 0,
        }

    def _check_files_in_db(self, start_candidate: SpatioTemporalPoint) -> bool:
        """Initializes the arena and downloads the data. Catches errors if download fails i.e. due to missing or corrupt files."""
        # derive t_interval from start_candidate
        t_interval = [start_candidate.date_time,
                      start_candidate.date_time + datetime.timedelta(seconds=self.config["time_horizon"])]
        try:
                self.arena = ArenaFactory.create(
                    scenario_config=self.config["scenario_config"],
                    x_interval=[units.Distance(deg=x) for x in self.config["x_range"]],
                    y_interval=[units.Distance(deg=y) for y in self.config["y_range"]],
                    t_interval=t_interval,
                    throw_exceptions=True,
                    c3=self.c3
                )
                self.arena.reset(PlatformState.from_spatio_temporal_point(start_candidate))
                return True
        except (MissingOceanFileException, CorruptedOceanFileException) as e:
            self.logger.warning(
                f"Aborted because of missing or corrupted files: [{self.config['t_range'][0]}, {self.config['t_range'][1]}]."
            )
            self.logger.warning(e)
            return False

    def generate_batch(self) -> List[SeaweedProblem]:
        """Generate a batch of starts according to config"""
        self.problems = []

        with timing_dict(
            self.performance,
            "total_time",
            "Batch finished ({})",
            self.logger,
        ):
            # Step 1: Generate start candidates
            starts = self._generate_starts()

            # Step 2: Create Problems
            for rand, start, distance_to_shore in starts:
                self.problems.append(
                    SeaweedProblem(
                        start_state=PlatformState(
                            lon=start.lon,
                            lat=start.lat,
                            date_time=start.date_time,
                        ),
                        platform_dict=self.arena.platform.platform_dict,
                        extra_info={
                            "time_horizon_in_sec": self.config["time_horizon"],
                            "random": rand,
                            "distance_to_shore_deg": distance_to_shore.deg,
                            # Factory
                            "factory_seed": self.config["seed"],
                            "factory_index": len(self.problems),
                        },
                    )
                )

        return self.problems

    def _generate_starts(self) -> List[Tuple[bool, SpatioTemporalPoint, float]]:
        """
        Samples in/feasible starts from the already generated target.
        Returns:
            List of starts for the pre-existing target.
        """
        start_time = time.time()
        sampled_points = []
        i = 0
        while i < self.config["n_samples"]:
            print("Iteration: ", i)
            point = SpatioTemporalPoint(
                lon=units.Distance(
                    deg=self.random.uniform(self.config["x_range"][0], self.config["x_range"][1])
                ),
                lat=units.Distance(
                    deg=self.random.uniform(self.config["y_range"][0], self.config["y_range"][1])
                ),
                date_time=self._get_random_starting_time(),
            )
            print(point)
            # check if all files are there and if so add to list
            if self._check_files_in_db(point):
                print("files is in db")
                # Add if far enough from shore
                distance_to_shore = self.arena.ocean_field.hindcast_data_source.distance_to_land(
                    point.to_spatial_point()
                )
                if distance_to_shore.deg > self.config["min_distance_from_land"]:
                    print("distance good")
                    sampled_points.append((True, point, distance_to_shore))
                    i += 1
                else:
                    print("distance too close to shore")


        self.logger.info(
            f"SeaweedMissionGenerator: {len(sampled_points)} starts created ({time.time()-start_time:.1f}s)."
        )

        return sampled_points

    def _get_random_starting_time(self):
        """Samples a random datetime object within the given time interval and ensures it is at least time_horizon earlier than the end of our time interval
        Returns:
            random_datetime
        """
        start_datetime, end_datetime = self.config["t_range"]
        end_datetime -= datetime.timedelta(seconds=self.config["time_horizon"])

        if end_datetime <= start_datetime:
            raise ValueError(
                "Invalid time interval, the time horizon is too large for the given time range"
            )

        random_seconds = random.uniform(0, (end_datetime - start_datetime).total_seconds())
        random_datetime = start_datetime + datetime.timedelta(seconds=random_seconds)

        return random_datetime

    @staticmethod
    def plot_starts(
        config: dict,
        problems: Optional[List] = None,
        results_folder: Optional[str] = None,
    ):

        # Step 1: Load Problems
        if problems is not None:
            problems_df = pd.DataFrame([problem.to_dict() for problem in problems])
        elif results_folder is not None:
            if results_folder.startswith("/seaweed-storage/"):
                cluster_utils.ensure_storage_connection()
            problems_df = pd.read_csv(f"{results_folder}problems.csv")
            analysis_folder = f"{results_folder}analysis/"
            os.makedirs(analysis_folder, exist_ok=True)
        else:
            raise ValueError(
                "Please provide a List of problems or a directory where the problems are saved in a .csv file as parameters."
            )
        problem = SeaweedProblem.from_pandas_row(problems_df.iloc[0])

        # Step 2:
        arena = ArenaFactory.create(
            scenario_file=config.get("scenario_file", None),
            scenario_config=config.get("scenario_config", {}),
            scenario_name=config.get("scenario_name", None),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
                x_interval=config["x_range"],
                y_interval=config["y_range"],
                time=problem.start_state.date_time,
                spatial_resolution=0.2,
                return_ax=True,
                # figsize=(32, 24),
            )
        ax.scatter(
            problems_df["x_0_lon"], problems_df["x_0_lat"], c="red", marker="o", s=6, label="starts"
        )

        ax.legend()
        if results_folder is not None:
            ax.get_figure().savefig(f"{analysis_folder}starts.png")
        ax.get_figure().show()


class SeaweedMissionGeneratorFeasibility:
    """
    The generator samples starting points (n_samples) within a given spatio-temporal region under some constraints (i.e. certain distance to land) and a given time horizon.
    """

    # TODO: test
    def __init__(self, config: dict = {}, c3=None):
        """Takes the config file, sets the seed and initializes the arena object.

        Args:
            config: dictionary which specifies the generation configuration
        """
        if type(config["t_range"][0]) == str:
            config["t_range"] = [datetime.datetime.fromisoformat(t) for t in config["t_range"]]
        self.config = config
        self.c3 = c3

        self.random = np.random.default_rng(self.config["seed"])

        self.logger = logging.getLogger("SeaweedMissionGenerator")

        self._initialize_arena_download_data()
        self.config["hj_specific_settings"].update({"platform_dict": self.config["platform_dict"]})

        self.planner = HJBSeaweed2DPlanner(
            arena=self.arena,
            problem=None,
            specific_settings=self.config["hj_specific_settings"],
            c3=self.c3,
        )

        if not hasattr(self, "arena"):
            raise AttributeError("Arena is not implemented")

    def _initialize_arena_download_data(self):
        """Initializes the arena and downloads the data. Catches errors if download fails i.e. due to missing or corrupt files."""

        self.performance = {
            "total_time": 0,
            "generate_time": 0,
            "errors": 0,
        }

        self.errors = []
        try:
            self.arena = ArenaFactory.create(
                scenario_file=self.config.get("scenario_file", None),
                scenario_config=self.config.get("scenario_config", {}),
                scenario_name=self.config.get("scenario_name", None),
                x_interval=[units.Distance(deg=x) for x in self.config["x_range"]],
                y_interval=[units.Distance(deg=y) for y in self.config["y_range"]],
                t_interval=self.config["t_range"],
                throw_exceptions=True,
                c3=self.c3,
            )
        except (MissingOceanFileException, CorruptedOceanFileException) as e:
            self.logger.warning(
                f"Aborted because of missing or corrupted files: [{self.config['t_range'][0]}, {self.config['t_range'][1]}]."
            )
            self.logger.warning(e)
            self.performance["errors"] += 1
            self.errors.append(str(e))
            return False

    def generate_batch(self) -> List[SeaweedProblem]:
        """Generate a batch of starts according to config"""
        self.problems = []

        with timing_dict(
            self.performance,
            "total_time",
            "Batch finished ({})",
            self.logger,
        ):
            # Step 1: Generate starts
            starts = self._generate_starts()

            # Step 2: Create Problems
            for rand, start, distance_to_shore in starts:
                self.problems.append(
                    SeaweedProblem(
                        start_state=PlatformState(
                            lon=start.lon,
                            lat=start.lat,
                            date_time=start.date_time,
                        ),
                        platform_dict=self.arena.platform.platform_dict,
                        extra_info={
                            "time_horizon_in_sec": self.config["time_horizon"],
                            "random": rand,
                            "distance_to_shore_deg": distance_to_shore.deg,
                            # Factory
                            "factory_seed": self.config["seed"],
                            "factory_index": len(self.problems),
                        },
                    )
                )

        return self.problems

    def _generate_starts(self) -> List[Tuple[bool, SpatioTemporalPoint, float]]:
        """
        Samples in/feasible starts from the already generated target.
        Returns:
            List of starts for the pre-existing target.
        """
        start_time_wall_clock = time.time()
        sampled_points = []

        start_times = [self._get_random_starting_time() for _ in range(self.config["n_samples"])]

        for start_time in start_times:
            (
                all_values_global_flipped,
                _,
                x_grid_global,
                y_grid_global,
            ) = self.planner._get_value_fct_reach_times(
                t_interval=[
                    start_time,
                    start_time
                    + datetime.timedelta(
                        seconds=self.planner.specific_settings["T_goal_in_seconds"]
                    ),
                ],
                u_max=self.planner.specific_settings["platform_dict"]["u_max_in_mps"],
                dataSource=self.planner.specific_settings["dataSource"],
            )

            # Get all values which are lower than 0.8 on the first time step of our value fct. and set them to one
            # -> Those areas are feasible since the platform will stay in our Regin
            infeasible_region_mask = np.where(all_values_global_flipped[0] < 0.8, 1, 0)

            # Unfortunateley our unfeasible areas smoothly transition into the normal values
            # so we don't want to sample close to the edges of the platos (unfeasible sets) but with some distance
            # We can make use of Andreas bfs_min_distance function which gives as a distance map (to land or in our case to the platos) over a grid
            # In our simulator we use 0 for land hence we set the infeasible platos also to 0

            lat_rad = np.deg2rad(y_grid_global)
            lon_rad = np.deg2rad(x_grid_global)

            min_d_map_km = self._bfs_min_distance(
                data=infeasible_region_mask, lat=lat_rad, lon=lon_rad
            )

            # We don't want to sample in a distance of 3 degree which is (1deg is between 85-111km in our Region) so at least 255km distance
            feasible_map = np.where(min_d_map_km > 255, 1, 0)

            while True:
                # Sample random lon lat coords.
                lon_sample = units.Distance(
                    deg=self.random.uniform(self.config["x_range"][0], self.config["x_range"][1])
                )
                lat_sample = units.Distance(
                    deg=self.random.uniform(self.config["y_range"][0], self.config["y_range"][1])
                )

                # Get the indices of the sample on the global grid
                lon_idx = self._get_closest_idx(x_grid_global, lon_sample)
                lat_idx = self._get_closest_idx(y_grid_global, lat_sample)

                # Check if sample is feasible and if yes add sample if also distant enough from land/shore
                if feasible_map[lon_idx, lat_idx] == 1:
                    point = SpatioTemporalPoint(
                        lon=lon_sample,
                        lat=lat_sample,
                        date_time=start_time,
                    )
                    # Add if far enough from shore
                    distance_to_shore = (
                        self.arena.ocean_field.hindcast_data_source.distance_to_land(
                            point.to_spatial_point()
                        )
                    )
                    if distance_to_shore.deg > self.config[
                        "min_distance_from_land"
                    ] and self._point_in_pacific(point):
                        sampled_points.append((True, point, distance_to_shore))
                        break

        self.logger.info(
            f"SeaweedMissionGenerator: {len(sampled_points)} starts created ({time.time()-start_time_wall_clock:.1f}s)."
        )

        return sampled_points

    def _get_random_starting_time(self):
        """Samples a random datetime object within the given time interval and ensures it is at least time_horizon earlier than the end of our time interval
        Returns:
            random_datetime
        """
        start_datetime, end_datetime = self.config["t_range"]
        end_datetime -= datetime.timedelta(seconds=self.config["time_horizon"])

        if end_datetime <= start_datetime:
            raise ValueError(
                "Invalid time interval, the time horizon is too large for the given time range"
            )

        random_seconds = random.uniform(0, (end_datetime - start_datetime).total_seconds())
        random_datetime = start_datetime + datetime.timedelta(seconds=random_seconds)

        return random_datetime

    @staticmethod
    def plot_starts(
        config: dict,
        problems: Optional[List] = None,
        results_folder: Optional[str] = None,
    ):

        # Step 1: Load Problems
        if problems is not None:
            problems_df = pd.DataFrame([problem.to_dict() for problem in problems])
        elif results_folder is not None:
            if results_folder.startswith("/seaweed-storage/"):
                cluster_utils.ensure_storage_connection()
            problems_df = pd.read_csv(f"{results_folder}problems.csv")
            analysis_folder = f"{results_folder}analysis/"
            os.makedirs(analysis_folder, exist_ok=True)
        else:
            raise ValueError(
                "Please provide a List of problems or a directory where the problems are saved in a .csv file as parameters."
            )
        problem = SeaweedProblem.from_pandas_row(problems_df.iloc[0])

        # Step 2:
        arena = ArenaFactory.create(
            scenario_file=config.get("scenario_file", None),
            scenario_config=config.get("scenario_config", {}),
            scenario_name=config.get("scenario_name", None),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
                x_interval=config["x_range"],
                y_interval=config["y_range"],
                time=problem.start_state.date_time,
                spatial_resolution=0.2,
                return_ax=True,
                # figsize=(32, 24),
            )
        ax.scatter(
            problems_df["x_0_lon"], problems_df["x_0_lat"], c="red", marker="o", s=6, label="starts"
        )

        ax.legend()
        if results_folder is not None:
            ax.get_figure().savefig(f"{analysis_folder}starts.png")
        ax.get_figure().show()

    def _point_in_pacific(self, point: SpatioTemporalPoint) -> bool:
        """Returns True if point in Region 3 is in Pacific, if in carribean sea returns False"""
        return not (
            (-90 < point.lon.deg < -70 and 15 < point.lat.deg < 20)
            or (-100 < point.lon.deg < -90 and 17.5 < point.lat.deg < 20)
            or (-85 < point.lon.deg < -70 and 10 < point.lat.deg < 15)
            or (-86 < point.lon.deg < -70 and 8.5 < point.lat.deg < 10)
        )

    def _bfs_min_distance(
        self, data: np.ndarray, lat: np.ndarray, lon: np.ndarray, connectivity: int = 4
    ) -> np.ndarray:
        """Breadth first search of minimum ditance to land. Uses geodetic distance.
        --> Not optimal, as the nonuniform distance steps can lead to a large difference of neighboring cells
        that are reached at the same step, but because they were visited from different directions,
        their values may differ by a very large margin (2k vs. 2.9k km,).
        Credits to: Andreas
        Args:
            data (np.ndarray): Elevation in km, nxm
            lat (np.ndarray): Latitude, n
            lon (np.ndarray): Longitude, m
            connectivity (int, optional): 4-connectivity: lool up neighboring cells, 8 connectivity look up diagonal neighbors as well. Defaults to 4.
        Raises:
            NotImplementedError: Only 4 and 8 connectivity implemented.
        Returns:
            np.ndarray: Distance in km to closest land.
        """
        if connectivity == 4:
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif connectivity == 8:
            dirs = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
        else:
            raise NotImplementedError()

        min_d_map = data.astype(float)
        visited = set()

        q = Buffer()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if min_d_map[i][j] == 0:  # land
                    q.append((i, j))

        while q:
            # First iteration will do all land, next ones will do all coastline, then all coastline + 1 field away
            for idx in range(len(q)):
                pos = q.popleft()
                i, j = pos
                visited.add(pos)

                distances = []
                for dir in dirs:
                    i_neighbor, j_neighbor = i + dir[0], j + dir[1]

                    # # Wraparound from -180 to 180 degrees
                    # if j_neighbor == data.shape[1]:
                    #     j_neighbor = 0
                    # if j_neighbor == -1:
                    #     j_neighbor = data.shape[1] - 1

                    # Add not visited neighboring sea cells to deque
                    # Add distance to nearest land
                    if (
                        i_neighbor >= 0
                        and i_neighbor < data.shape[0]
                        and j_neighbor >= 0
                        and j_neighbor < data.shape[1]
                    ):
                        # Add sea neighbors
                        if (i_neighbor, j_neighbor) not in visited and min_d_map[i_neighbor][
                            j_neighbor
                        ] != 0:
                            if not q.contains((i_neighbor, j_neighbor)):
                                q.append((i_neighbor, j_neighbor))

                        # Skip finding land if we are on land
                        if min_d_map[i][j] == 0:
                            continue
                        distances.append(
                            haversine_distances(
                                [[lon[i], lat[j]]], [[lon[i_neighbor], lat[j_neighbor]]]
                            )[0][0]
                            + min_d_map[i_neighbor][j_neighbor]
                        )
                if min_d_map[i][j] != 0:
                    min_d_map[i][j] = min(distances)

        min_d_map *= 6371  # Convert to kilometers

        return min_d_map

    def _get_closest_idx(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value.deg)).argmin()
        return idx


class Buffer:
    """Deque buffer with fast insert, popleft, remove and lookup.
    Deque has very slow lookup (elem in deque) that gets slower with more elements.
    """

    def __init__(self):
        self.queue = deque()
        self.value_set = set()

    def append(self, value):
        if self.contains(value):
            return
        self.queue.append(value)
        self.value_set.add(value)

    def contains(self, value):
        return value in self.value_set

    def popleft(self):
        value = self.queue.popleft()
        self.value_set.remove(value)
        return value

    def __len__(self):
        return len(self.value_set)
