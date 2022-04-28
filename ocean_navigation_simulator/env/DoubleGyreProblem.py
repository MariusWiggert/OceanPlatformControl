import dataclasses
import math

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

    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        ax.scatter(self.start_state.lon.deg, self.start_state.lat.deg, facecolors='none', edgecolors='red', marker='o', s=50, label='start')
        ax.add_patch(plt.Circle((self.end_region.lon.deg, self.end_region.lat.deg), 0.1, facecolor='none', edgecolor='green', label='goal'))

        return ax