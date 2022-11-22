import math

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction


class NaiveController(Controller):
    """
    Naive to Target, Full-power Actuation towards the goal (meant as a baseline)
    """

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
