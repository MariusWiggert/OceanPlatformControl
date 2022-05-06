import math
import numpy as np

from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformAction
from ocean_navigation_simulator.env.controllers.Controller import Controller


class NaiveToTargetController(Controller):
    """
    Naive to Target, Full-power Actuation towards the goal (meant as a baseline)
    """

    def __init__(self, problem: Problem):
        """
        StraightLineController constructor
        Args:
            problem: the Problem the controller will run on
        """
        super().__init__(problem)

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """
        Return action that goes in the direction of the target with full power.
        Args:
            observation: observation returned by the simulator
        Returns:
            SimulatorAction dataclass
        """
        # TODO: change how this functions for complex end regions (pretend it's a state for now)

        # Calculate the delta lon, delta lat, and magnitude (for normalization)
        dlon = self.problem.end_region.lon.deg - observation.platform_state.lon.deg
        dlat = self.problem.end_region.lat.deg - observation.platform_state.lat.deg
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go towards the center of the target with full power
        return PlatformAction.from_xy_propulsion(x_propulsion=dlon / mag, y_propulsion=dlat / mag)
