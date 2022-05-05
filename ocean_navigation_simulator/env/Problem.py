import abc
import dataclasses
from typing import Dict

import matplotlib.axes

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

    @abc.abstractmethod
    def is_done(self, state: PlatformState) -> bool:
        """
        Checks whether the problem is solved. Needs to get the current
        platform state.
        Args:
            state: PlatformState
        Returns:
            bool
        """
        pass

    @abc.abstractmethod
    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """
        Plots a specific problem on a map using a given maplotlib axis. The
        Idea here is that each problem might have a specific way of
        illustration. Plotting the problem in the problem class allows to
        have tailored illustrations.
        Returns:
            Matplot lib Axes
        """
        pass
