import abc
import dataclasses
from typing import Dict

import matplotlib.axes
import matplotlib.pyplot as plt

from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.data_sources.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.environment.data_sources.DataSources import DataSource


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

    # obstacle_regions: None = None  # TODO
    # config: Dict = {}  # TODO

    # @abc.abstractmethod
    def is_done(self) -> bool:
        """
        Yield the next problem to be used by the Gym environment.
        Returns:
            Next problem as a Problem object
        """
        pass

    def plot(self, ax: matplotlib.axes.Axes = None, data_source: DataSource = None,
             margin: float = 1, return_ax: bool = False) -> matplotlib.axes.Axes:
        """Helper Function to plot the problem on an existing ax."""
        if data_source:
            _, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
                x_0=self.start_state.to_spatio_temporal_point(),
                x_T=self.end_region, deg_around_x0_xT_box=margin, temp_horizon_in_s=0)
            ax = data_source.plot_data_at_time_over_area(time=self.start_state.date_time,
                                                         x_interval=x_interval, y_interval=y_interval, ax=ax,
                                                         return_ax=True)

        if ax is None and data_source is None:
            ax = plt.axes()

        ax.scatter(self.start_state.lon.deg, self.start_state.lat.deg, c='r', marker='o', s=100, label='start')
        goal_circle = plt.Circle((self.end_region.lon.deg, self.end_region.lat.deg),
                                 self.target_radius, color='g', fill=True, alpha=0.6, label='target')
        ax.add_patch(goal_circle)
        ax.legend(loc='upper right')
        if return_ax:
            return ax
        else:
            plt.show()
