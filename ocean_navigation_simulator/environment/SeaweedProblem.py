import dataclasses
import datetime
from typing import Optional

import matplotlib

from ocean_navigation_simulator.environment.PlatformState import PlatformState
from ocean_navigation_simulator.utils import units

# TODO: add minimal docstrings and comments for others to build on it!


@dataclasses.dataclass
class SeaweedProblem:
    """class to hold the essential variables for a path planning problem (A -> B)"""

    start_state: PlatformState
    platform_dict: dict = None
    extra_info: dict = dataclasses.field(default_factory=dict)

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        problem_start_color: Optional[str] = "red",
    ) -> matplotlib.axes.Axes:
        """plot start/target on a given axis"""
        ax.scatter(
            self.start_state.lon.deg,
            self.start_state.lat.deg,
            c=problem_start_color,
            marker="o",
            label="start",
        )

        return ax

    def is_done(self, state: PlatformState) -> bool:
        """Checks whether the problem is solved or became unsolvable. Needs to get the current
        platform state.
        Args:
            state: PlatformState
        Returns:
          1: problem successfully solved
          0: problem not yet solved
          #-1: problem cannot be solved anymore (e.g. timeout)
        """
        return 0

    @staticmethod
    def from_pandas_row(mission):
        return SeaweedProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=mission["x_0_lon"]),
                lat=units.Distance(deg=mission["x_0_lat"]),
                date_time=datetime.datetime.fromisoformat(mission["t_0"]),
            ),
            extra_info=mission.to_dict() | {"index": mission.name},
        )

    def to_dict(self) -> dict:
        return {
            "t_0": self.start_state.date_time.isoformat(),
            "x_0_lon": self.start_state.lon.deg,
            "x_0_lat": self.start_state.lat.deg,
        } | (self.extra_info if self.extra_info is not None else {})

    def to_c3_mission_config(self):
        """To easily populate c3 database with missions."""
        prob_dict = self.to_dict()
        x_0 = {
            "lon": prob_dict["x_0_lon"],
            "lat": prob_dict["x_0_lat"],
            "date_time": prob_dict["t_0"],
        }

        mission_config = {
            "x_0": [x_0],  # Evaluation runner assumes it is a list (for multi-agent)
            "seed": prob_dict.get("factory_seed", None),
        }

        return mission_config

    @staticmethod
    def from_c3_mission_config(missionConfig):
        return SeaweedProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=missionConfig["x_0"][0]["lon"]),
                lat=units.Distance(deg=missionConfig["x_0"][0]["lat"]),
                date_time=datetime.datetime.fromisoformat(missionConfig["x_0"][0]["date_time"]),
            ),
        )

    def __repr__(self):
        return "Problem [start: {s}]".format(
            s=self.start_state.to_spatio_temporal_point(),
        )
