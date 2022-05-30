import datetime
import math

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.controllers.Controller import Controller


class NaiveToTargetController(Controller):
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

    def get_waypoints(self) -> list:
        """
        Output start and end waypoints for the planner.
        Returns:
            List of format [start, end], where both start and end are of format [lat, lon, time]
        """
        start = [self.problem.start_state.lat, self.problem.start_state.lon, self.problem.start_state.date_time]

        # TODO: change how this functions for complex end regions
        end = [self.problem.end_region.lat, self.problem.end_region.lon,
               self.problem.start_state.date_time + datetime.timedelta(days=10)]
        return [start, end]
