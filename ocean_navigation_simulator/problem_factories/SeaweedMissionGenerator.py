import datetime
import logging
import os
import random
import time
from typing import List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatioTemporalPoint

from ocean_navigation_simulator.environment.SeaweedProblem import SeaweedProblem
from ocean_navigation_simulator.utils import cluster_utils, units


class SeaweedMissionGenerator:
    """
    The generator samples starting points (n_samples) within a given spatio-temporal region under some constraints (i.e. certain distance to land) and a given time horizon.
    """

    # TODO: test
    def __init__(self, config: dict = {}):
        """Takes the config file, sets the seed and initializes the arena object.

        Args:
            config: dictionary which specifies the generation configuration
        """
        if type(config["t_range"][0]) == str:
            config["t_range"] = [datetime.datetime.fromisoformat(t) for t in config["t_range"]]
        self.config = config

        self.random = np.random.default_rng(self.config["seed"])

        self.arena = ArenaFactory.create(
            scenario_file=self.config.get("scenario_file", None),
            scenario_config=self.config.get("scenario_config", {}),
            scenario_name=self.config.get("scenario_name", None),
        )

        self.logger = logging.getLogger("SeaweedMissionGenerator")

    def generate_batch(self) -> List[SeaweedProblem]:
        """Generate a batch of starts according to config"""
        self.problems = []

        # Step 1: Generate starts
        starts = self._generate_starts()

        # Step 2: Create Problems
        for random, start, distance_to_shore in starts:
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
                        "random": random,
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
    def plot_starts_and_targets(
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
