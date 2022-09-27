import abc
import logging

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Platform import PlatformAction


class Controller(abc.ABC):
    """
    Interface for controllers.
    """

    gpus: float = 0.0

    def __init__(self, problem: NavigationProblem):
        """
        Basic constructor logging the problem given at construction.
        Args:
            problem: the Problem the controller will run on
        """
        self.problem = problem
        # initialize logger
        self.logger = logging.getLogger("arena.controller")
        self.logger.setLevel(logging.INFO)

    @abc.abstractmethod
    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """Given an observation, outputs the controller's next action
        Args:
          observation: observed state from simulator or other source (i.e. observer, other controller)
        Returns:
          Controller's next action as a PlatformAction object.
        """
