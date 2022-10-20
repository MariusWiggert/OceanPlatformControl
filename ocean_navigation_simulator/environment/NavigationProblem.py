import dataclasses
import datetime
from typing import Optional

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
class NavigationProblem(Problem):
    """class to hold the essential variables for a path planning problem (A -> B)"""

    start_state: PlatformState
    end_region: SpatialPoint
    target_radius: float
    timeout: datetime.timedelta = None
    platform_dict: dict = None

    def passed_seconds(self, state: PlatformState) -> float:
        return (state.date_time - self.start_state.date_time).total_seconds()

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
            -1  if problem timed out
        """
        if self.passed_seconds(state) >= self.timeout.total_seconds():
            return -1
        elif state.distance(self.end_region).deg <= self.target_radius:
            return 1
        return 0

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        problem_start_color: Optional[str] = "red",
        problem_target_color: Optional[str] = "green",
    ) -> matplotlib.axes.Axes:
        """plot start/target on a given axis"""
        ax.scatter(
            self.start_state.lon.deg,
            self.start_state.lat.deg,
            c=problem_start_color,
            marker="o",
            label="start",
        )
        ax.add_patch(
            plt.Circle(
                (self.end_region.lon.deg, self.end_region.lat.deg),
                self.target_radius,
                facecolor=problem_target_color,
                edgecolor=problem_target_color,
                label="goal",
            )
        )

        return ax

    def __repr__(self):
        return "Problem [start: {s}, end: {e}, target_radius: {r:.2f}, timeout: {t:.1f}h]".format(
            s=self.start_state.to_spatio_temporal_point(),
            e=self.end_region,
            r=self.target_radius,
            t=self.timeout.total_seconds() / 3600,
        )
