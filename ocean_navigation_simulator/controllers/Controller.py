import abc
from typing import Optional

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.Problem import Problem

class Controller(abc.ABC):
    """
    Interface for controllers.
    """
    gpus: float = 0.0

    def __init__(self, problem: NavigationProblem, verbose: Optional[int] = 0, **kwargs):
        """
        Basic constructor logging the problem given at construction.
        Args:
            problem: the Problem the controller will run on
        """
        self.problem = problem
        self.verbose = verbose

    @abc.abstractmethod
    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """ Given an observation, outputs the controller's next action
        Args:
          observation: observed state from simulator or other source (i.e. observer, other controller)
        Returns:
          Controller's next action as a PlatformAction object.
        """