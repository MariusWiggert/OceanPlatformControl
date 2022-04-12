from ocean_navigation_simulator.env.problem import Problem
from ocean_navigation_simulator.env.simulator_data import SimulatorObservation, SimulatorAction
from ocean_navigation_simulator.env.controllers.controller import Controller
from ocean_navigation_simulator.env.controllers.utils import transform_u_dir_to_u
import math
import numpy as np


class NaiveToTargetController(Controller):
    """
    Straight Line, Full-power Actuation towards the goal (meant as a baseline)
    """

    def __init__(self, problem: Problem):
        """
        StraightLineController constructor
        Args:
            problem: the Problem the controller will run on
        """
        self.problem = problem
        self.start_state = problem.start_state
        self.end_region = problem.end_region

    def get_action(self, observation: SimulatorObservation) -> SimulatorAction:
        """
        Return action that goes in the direction of the target with full power.
        Args:
            observation: observation returned by the simulator
        Returns:
            SimulatorAction dataclass
        """
        current_state = observation.platform_observation

        lon, lat = current_state.lon, current_state.lat

        # TODO: change how this functions for complex end regions (pretend it's a state for now)
        lon_target, lat_target = self.end_region.lon, self.end_region.lat

        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go there full power
        u_dir = np.array([[dlon / mag], [dlat / mag]])
        thrust, heading = transform_u_dir_to_u(u_dir=u_dir)

        return SimulatorAction(thrust, heading)

    def get_waypoints(self) -> list:
        """
        Output start and end waypoints for the planner.
        Returns:
            List of format [start, end], where both start and end are of format [lat, lon, time]
        """
        start = [self.start_state.lat, self.start_state.lon, self.start_state.date_time]

        # TODO: change how this functions for complex end regions
        end = [self.end_region.lat, self.end_region.lon, self.end_region.date_time]
        return [start, end]
