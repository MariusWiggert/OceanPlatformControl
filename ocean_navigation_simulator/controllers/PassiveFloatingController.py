from typing import Dict, Optional
from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.SeaweedProblem import SeaweedProblem
from c3python import C3Python


class PassiveFloatController(Controller):
    """
    Passively floating controller, always puts out a no propulsion action
    """

    def __init__(
        self,
        arena: Arena,
        problem: SeaweedProblem,
        specific_settings: Optional[Dict] = ...,
        c3: Optional[C3Python] = None,
    ):

        # get arena object for accessing seaweed growth model
        self.arena = arena
        self.c3 = c3
        super().__init__(problem, specific_settings)

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        return PlatformAction.from_xy_propulsion(x_propulsion=0, y_propulsion=0)
