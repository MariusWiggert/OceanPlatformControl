from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.controllers.NaiveSafetyController import NaiveSafetyController
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)

# Ugly import to enable importing all controllers using eval
import ocean_navigation_simulator.controllers


class SwitchingController(Controller):
    def __init__(
        self,
        problem: NavigationProblem,
        specific_settings: dict,
        specific_settings_navigation,
        specific_settings_safety,
    ) -> None:
        super().__init__(problem)

        self.specific_settings = specific_settings
        self.specific_settings_navigation = specific_settings_navigation
        self.specific_settings_safety = specific_settings_safety
        # Eval to execute and import the controller we want to prevent a long switch case with more imports.
        # TODO: discuss method
        self.navigation_controller = eval(self.specific_settings["navigation_controller"])
        self.safety_controller = eval(self.specific_settings["safety_controller"])
        self.safety_status = False

    def safety_condition(self, observation: ArenaObservation) -> bool:
        # If True, switch to safety
        if self.specific_settings["safety_condition"] == "distance":
            distance_to_land_at_pos = self.safety_controller.distance_map.interp(
                lon=observation.platform_state.lon.deg, lat=observation.platform_state.lat.deg
            )["distance"].data
            return distance_to_land_at_pos < self.specific_settings["safe_distance_to_land"]
        elif self.specific_settings["safety_condition"] == "on":
            return True
        elif self.specific_settings["safety_condition"] == "off":
            return False
        elif self.specific_settings["safety_condition"] == "distance_and_time_safe":
            # TODO: implement time counter
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        # Get the action depending on switching condition
        safety_status = self.safety_condition(observation)
        if safety_status != self.safety_status:
            print(f"SwitchingController: Safety switched to {safety_status}")
            self.logger.debug(f"SwitchingController: Safety switched to {safety_status}")
            self.safety_status = safety_status

        if safety_status:
            return self.safety_controller.get_action(observation)
        else:
            return self.navigation_controller.get_action(observation)
