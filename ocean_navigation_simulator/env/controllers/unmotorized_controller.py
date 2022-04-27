import math
import numpy as np

from ocean_navigation_simulator.env.problem import Problem
from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformAction
from ocean_navigation_simulator.env.controllers.controller import Controller

class UnmotorizedController(Controller):
    """
    No action, the platform just follows the current
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

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """
        Return action that goes in the direction of the target with full power.
        Args:
            observation: observation returned by the simulator
        Returns:
            SimulatorAction dataclass
        """
        return PlatformAction(magnitude=0, direction=0)

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
