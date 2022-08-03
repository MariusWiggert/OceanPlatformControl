import dataclasses
from datetime import datetime
from typing import Optional

import matplotlib
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.utils import units


@dataclasses.dataclass
class NavigationProblem(Problem):
    start_state: PlatformState
    end_region: SpatialPoint
    target_radius: float
    timeout: float = None # TODO: implement timeout

    def passed_seconds(self, state: PlatformState) -> float:
        return (state.date_time - self.start_state.date_time).total_seconds()

    def is_done(self, state: PlatformState) -> int:
        if self.passed_seconds(state) >= self.timeout:
            return -1
        elif state.distance(self.end_region) <= self.target_radius:
            return 1
        return 0

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        color: Optional[str] = 'green',
    ) -> matplotlib.axes.Axes:
        ax.scatter(self.start_state.lon.deg, self.start_state.lat.deg, facecolors='none', edgecolors=color, marker='o', label='start')
        ax.add_patch(plt.Circle((self.end_region.lon.deg, self.end_region.lat.deg), self.target_radius, facecolor='none', edgecolor=color, label='goal'))

        return ax

    @staticmethod
    def from_mission(mission):
        return NavigationProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=mission['x_0_lon']),
                lat = units.Distance(deg=mission['x_0_lat']),
                date_time = datetime.fromisoformat(mission['t_0'])
            ),
            end_region = SpatialPoint(
                lon=units.Distance(deg=mission['x_T_lon']),
                lat=units.Distance(deg=mission['x_T_lat'])
            ),
            target_radius = 0.1,
            timeout = 5 * 24 * 3600
        )