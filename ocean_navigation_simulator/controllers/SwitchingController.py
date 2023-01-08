from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformAction

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.controllers.NaiveSafetyController import NaiveSafetyController

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

    def safety_condition(self, observation: ArenaObservation) -> str:
        """Compute safety status of platform, depening on safety_condition provided by config.
        Can return False: no safety action required, or returns the string of the area_type, which area to avoid.
        Returns first area_type that violates safe distance.

        Args:
            observation (ArenaObservation): Environment status.

        Raises:
            NotImplementedError: In case we have not implemented a certain status.

        Returns:
            str: Safety status to take.
        """
        if self.specific_settings["safety_condition"]["base_setting"] == "off":
            return False
        elif self.specific_settings["safety_condition"]["base_setting"] == "always_on":
            return self.specific_settings["safety_condition"]["area_type"][0]
        elif self.specific_settings["safety_condition"]["base_setting"] == "on":
            for area_type in self.specific_settings["safety_condition"]["area_type"]:
                distance_to_area_at_pos = (
                    self.safety_controller.distance_map[area_type]
                    .interp(
                        lon=observation.platform_state.lon.deg,
                        lat=observation.platform_state.lat.deg,
                    )["distance"]
                    .data
                )
                if distance_to_area_at_pos < self.specific_settings["safe_distance"][area_type]:
                    return area_type
            return False
        else:
            raise NotImplementedError

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """Get action, safety_status from safety_condition determines which controller provides action.

        Args:
            observation (ArenaObservation): Environment state.

        Returns:
            PlatformAction: Action to take.
        """
        safety_status = self.safety_condition(observation)
        if safety_status != self.safety_status:
            print(f"SwitchingController: Safety switched to {safety_status}")
            self.logger.debug(f"SwitchingController: Safety switched to {safety_status}")
            self.safety_status = safety_status

        if safety_status:
            return self.safety_controller.get_action(observation, area_type=safety_status)
        else:
            return self.navigation_controller.get_action(observation)
