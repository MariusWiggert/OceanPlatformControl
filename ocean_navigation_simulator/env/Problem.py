import abc
import dataclasses
from typing import Dict

import matplotlib.axes
import matplotlib.pyplot as plt

from ocean_navigation_simulator.env.Platform import PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint

@dataclasses.dataclass
class Problem(abc.ABC):
    """
    A path planning problem for a Planner to solve.

    Attributes:
        start_state:
            PlatformState object specifying where the agent will start
        end_region:
            [UndefinedObject] specifying the region the agent should try reach (positive reward)
        obstacle_regions:
            [UndefinedObject] specifying the region(s) the agent should avoid (negative reward)
        config:
            A dict specifying the platform/problem parameters
            TODO: make this config into class?
    """
    start_state: PlatformState
    end_region: SpatialPoint  # TODO
    target_radius: float
    #obstacle_regions: None = None  # TODO
    #config: Dict = {}  # TODO

    # @abc.abstractmethod
    def is_done(self) -> bool:
        """
        Yield the next problem to be used by the Gym environment.
        Returns:
            Next problem as a Problem object
        """
        pass

    # @abc.abstractmethod
    def plot_on_currents(self, ax: matplotlib.axes.Axes, data_source) -> matplotlib.axes.Axes:
        """Helper Function to plot the problem on an existing ax."""
        ax.scatter(self.start_state.lon.deg, self.start_state.lat.deg, c='r', marker='o', s=100, label='start')
        goal_circle = plt.Circle((self.end_region.lon.deg, self.end_region.lat.deg),
                                self.target_radius, color='g', fill=True, alpha=0.6, label='target')
        ax.add_patch(goal_circle)
        ax.legend(loc='upper right')
        return ax
