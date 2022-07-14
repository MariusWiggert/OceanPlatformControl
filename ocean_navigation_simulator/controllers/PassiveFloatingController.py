import datetime
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.controllers.Controller import Controller


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
