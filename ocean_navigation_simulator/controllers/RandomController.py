import numpy as np

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Platform import PlatformAction


class RandomController(Controller):
    """
    Naive to Target, Full-power Actuation towards the goal (meant as a baseline)
    """

    def __init__(self, problem: NavigationProblem, actions=None, seed=2022):
        super().__init__(problem)
        self.actions = actions
        self.random = np.random.default_rng(seed)

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """
        Return random direction with full power (discretised if actions given in constructor)
        Args:
            observation: observation returned by the simulator
        Returns:
            SimulatorAction dataclass
        """
        if self.actions is not None:
            direction = self.random.integers(self.actions) * 2 * np.pi / self.actions
        else:
            direction = self.random.uniform(0, 2 * np.pi)

        return PlatformAction(magnitude=1, direction=direction)
