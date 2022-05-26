import math
import numpy as np

from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformAction
from ocean_navigation_simulator.env.controllers.Controller import Controller


class PassiveFloatController(Controller):
    """
    Passively floating controller, always puts out a no propulsion action
    """

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """
        Return action that goes in the direction of the target with full power.
        Args:
            observation: observation returned by the simulator
        Returns:
            SimulatorAction dataclass
        """
        # return 0 always
        return PlatformAction.from_xy_propulsion(x_propulsion=0, y_propulsion=0)
