import abc
import numpy as np
from typing import Tuple, Optional, Dict
import bisect

from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformAction
from ocean_navigation_simulator.env.problem import Problem
from ocean_navigation_simulator.env.Platform import PlatformState

# TODO: other methods needed?


class Controller(abc.ABC):
    """
    Interface for controllers.
    """

    def __init__(self, problem: Problem, specific_settings: Optional[Dict] = None):
        """
        Basic constructor logging the problem given at construction.
        Args:
            problem: the Problem the controller will run on
        """
        self.problem = problem
        # Note: managing the forecast fieldsets is done in the simulator
        self.forecast_data_source = None
        self.updated_forecast_source = True

        # initialize vectors for open_loop control
        self.times, self.x_traj, self.contr_seq = None, None, None

        # saving the planned trajectories for inspection purposes
        self.planned_trajs = []

        self.specific_settings = specific_settings

    @abc.abstractmethod
    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """ Given an observation, outputs the controller's next action
        Args:
          observation: observed state from simulator or other source (i.e. observer, other controller)
        Returns:
          Controller's next action as a numpy array.
        """

    def get_open_loop_control_from_plan(self, state: PlatformState) -> PlatformAction:
        """ Indexing into the planned open_loop control sequence using the time from state.
        Args:
            state    PlatformState containing [lat, lon, battery_level, date_time]
        Returns:
            PlatformAction object
        """
        # an easy way of finding for each time, which index of control signal to apply
        idx = bisect.bisect_right(self.times, state.date_time.timestamp()) - 1
        if idx == len(self.times) - 1:
            idx = idx - 1
            print("Controller Warning: continuing using last control although not planned as such")

        # extract right element from ctrl vector
        return PlatformAction(magnitude=self.contr_seq[0, idx], direction=self.contr_seq[1, idx])

