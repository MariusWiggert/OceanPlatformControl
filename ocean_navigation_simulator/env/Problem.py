import abc
import dataclasses
from typing import Dict

import matplotlib.axes

from ocean_navigation_simulator.env.Platform import PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint


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
    #obstacle_regions: None = None  # TODO
    #config: Dict = {}  # TODO

    @abc.abstractmethod
    def is_done(self) -> bool:
        """
        Yield the next problem to be used by the Gym environment.
        Returns:
            Next problem as a Problem object
        """
        pass

    @abc.abstractmethod
    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        pass
