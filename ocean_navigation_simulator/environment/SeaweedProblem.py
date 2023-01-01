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

    @staticmethod
    def from_pandas_row(mission):
        return SeaweedProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=mission["x_0_lon"]),
                lat=units.Distance(deg=mission["x_0_lat"]),
                date_time=datetime.datetime.fromisoformat(mission["t_0"]),
            ),
            x_range=[
                units.Distance(deg=mission["x_range_l"]),
                units.Distance(deg=mission["x_range_h"]),
            ]
            if "x_range_l" in mission
            else None,
            y_range=[
                units.Distance(deg=mission["y_range_l"]),
                units.Distance(deg=mission["y_range_h"]),
            ]
            if "x_range_h" in mission
            else None,
            extra_info=mission.to_dict() | {"index": mission.name},
        )

    def to_dict(self) -> dict:
        return (
            {
                "t_0": self.start_state.date_time.isoformat(),
                "x_0_lon": self.start_state.lon.deg,
                "x_0_lat": self.start_state.lat.deg,
            }
            | (
                {
                    "x_range_l": self.x_range[0].deg,
                    "x_range_h": self.x_range[1].deg,
                }
                if self.x_range is not None
                else {}
            )
            | (
                {
                    "x_range_l": self.y_range[0].deg,
                    "x_range_h": self.y_range[1].deg,
                }
                if self.y_range is not None
                else {}
            )
            | (self.extra_info if self.extra_info is not None else {})
        )

    def __repr__(self):
        return "Problem [start: {s}]".format(
            s=self.start_state.to_spatio_temporal_point(),
        )
