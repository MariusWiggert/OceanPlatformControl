import abc
from typing import Optional, Dict

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.Problem import Problem

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

    @abc.abstractmethod
    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """ Given an observation, outputs the controller's next action
        Args:
          observation: observed state from simulator or other source (i.e. observer, other controller)
        Returns:
          Controller's next action as a PlatformAction object.
        """