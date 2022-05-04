import dataclasses
import math
from typing import Optional

import matplotlib
from matplotlib import pyplot as plt

from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.Problem import Problem

@dataclasses.dataclass
class DoubleGyreProblem(Problem):
    start_state: PlatformState
    end_region: SpatialPoint
    radius: float

    def is_done(self, state: PlatformState) -> bool:
        distance = state.distance(self.end_region)
        return distance <= self.radius

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        color: Optional[str] = 'green',
    ) -> matplotlib.axes.Axes:
        ax.scatter(self.start_state.lon.deg, self.start_state.lat.deg, facecolors='none', edgecolors=color, marker='o', s=50, label='start')
        ax.add_patch(plt.Circle((self.end_region.lon.deg, self.end_region.lat.deg), self.radius, facecolor='none', edgecolor=color, label='goal'))

        return ax