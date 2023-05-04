import dataclasses
import datetime
from typing import List, Optional

import matplotlib
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.utils import units

# TODO: add minimal docstrings and comments for others to build on it!


@dataclasses.dataclass
class NavigationWaypointsProblem(Problem):
    """class to hold the essential variables for a path planning problem (A -> B)"""

    start_state: PlatformState
    waypoints: List = None
    end_region: SpatialPoint
    target_radius: float
    platform_dict: dict = None
    x_range: List = None
    y_range: List = None
    extra_info: dict = dataclasses.field(default_factory=dict)

    def distance(self, state: PlatformState) -> units.Distance:
        return self.end_region.distance(state.to_spatial_point())

    def bearing(self, state: PlatformState) -> float:
        return self.end_region.bearing(state.to_spatial_point())

    def is_done(self, state: PlatformState) -> int:
        """
        Get the problem status
        Returns:
            1   if problem was solved
            0   if problem is still open
        """
        if state.distance(self.end_region).deg <= self.target_radius:
            return 1
        return 0

    def add_waypoints(self, waypoints_list) -> int:
        self.waypoints = waypoints_list
        return 1

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        problem_start_color: Optional[str] = "red",
        problem_target_color: Optional[str] = "green",
        start_size=100,
    ) -> matplotlib.axes.Axes:
        """plot start/target on a given axis"""
        ax.scatter(
            self.start_state.lon.deg,
            self.start_state.lat.deg,
            c=problem_start_color,
            marker="o",
            s=start_size,
            label="start",
            zorder=6,
        )
        ax.add_patch(
            plt.Circle(
                (self.end_region.lon.deg, self.end_region.lat.deg),
                self.target_radius,
                facecolor=problem_target_color,
                edgecolor=problem_target_color,
                label="goal",
                zorder=6,
            )
        )
        i = 0
        for point in self.waypoints:
            ax.add_patch(
            plt.Circle(
                (point[0], point[1]),
                self.target_radius,
                facecolor='yellow',
                edgecolor='yellow',
                label="waypoint" + str(i),
                zorder=6,
                )
            )
            i += 1


        return ax

    @staticmethod
    def from_pandas_row(mission):
        return NavigationProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=mission["x_0_lon"]),
                lat=units.Distance(deg=mission["x_0_lat"]),
                date_time=datetime.datetime.fromisoformat(mission["t_0"]),
            ),
            end_region=SpatialPoint(
                lon=units.Distance(deg=mission["x_T_lon"]),
                lat=units.Distance(deg=mission["x_T_lat"]),
            ),
            target_radius=mission["target_radius"],
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
                "x_T_lon": self.end_region.lon.deg,
                "x_T_lat": self.end_region.lat.deg,
                "target_radius": self.target_radius,
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
        return "Problem [start: {s}, end: {e}, target_radius: {r:.2f}]".format(
            s=self.start_state.to_spatio_temporal_point(), e=self.end_region, r=self.target_radius
        )