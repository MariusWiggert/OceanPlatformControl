import dataclasses
import datetime
import math
from typing import Optional

import matplotlib
from matplotlib import pyplot as plt
from numpy import Inf

from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.Problem import Problem

@dataclasses.dataclass
class NavigationProblem(Problem):
    start_state: PlatformState
    end_region: SpatialPoint
    target_radius: float
    timeout: float = None # TODO: implement timeout

    def is_done(self, state: PlatformState) -> int:
        time_diff = (state.date_time - self.start_state.date_time).total_seconds()
        distance = state.distance(self.end_region)

        if time_diff >= self.timeout:
            return -1
        elif distance <= self.target_radius:
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