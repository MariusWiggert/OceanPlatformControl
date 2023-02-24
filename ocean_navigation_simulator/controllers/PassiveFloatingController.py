from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction


class PassiveFloatingController(Controller):
    """
    Passively floating controller, always puts out a no propulsion action
    """

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        return PlatformAction.from_xy_propulsion(x_propulsion=0, y_propulsion=0)
