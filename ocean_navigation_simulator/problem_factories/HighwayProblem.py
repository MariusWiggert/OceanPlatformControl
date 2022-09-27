from dataclasses import dataclass
import math
from typing import Optional

import matplotlib
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.utils.units import Distance


@dataclass
class HighwayProblem(Problem):
    start_state: PlatformState
    end_region: SpatialPoint
    target_radius: Distance = Distance(meters=10)

    def is_done(self, state: PlatformState) -> bool:
        return state.distance(self.end_region) <= self.radius

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        color: Optional[str] = "green",
    ) -> matplotlib.axes.Axes:
        ax.scatter(
            self.start_state.lon.deg,
            self.start_state.lat.deg,
            facecolors="none",
            edgecolors=color,
            marker="o",
            s=50,
            label="start",
        )
        ax.add_patch(
            plt.plot(self.end_region.lon.deg, self.end_region.lat.deg, c=color, label="goal")
        )

        return ax
