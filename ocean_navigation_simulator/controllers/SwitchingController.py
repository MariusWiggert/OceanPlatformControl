from importlib import import_module

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformAction


class SwitchingController(Controller):
    def __init__(
        self,
        problem: NavigationProblem,
        specific_settings: dict,
        # specific_settings_navigation,
        # specific_settings_safety,
    ) -> None:
        super().__init__(problem)

        self.specific_settings = specific_settings["specific_settings_switching"]
        self.specific_settings_navigation = specific_settings["specific_settings_navigation"]
        self.specific_settings_safety = specific_settings["specific_settings_safety"]
        # Load module that is needed for the specific controller
        NavigationControllerClass = self.__import_class_from_string(
            self.specific_settings["navigation_controller"]
        )
        self.navigation_controller = NavigationControllerClass(
            problem, self.specific_settings_navigation
        )
        SafetyControllerClass = self.__import_class_from_string(
            self.specific_settings["safety_controller"]
        )
        self.safety_controller = SafetyControllerClass(problem, self.specific_settings_safety)
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

    def __import_class_from_string(self, controller_str: str):
        """Import a dotted module path from controller_str and return the attribute/class designated by the last name in the path. Raise ImportError if the import failed.
        Args:
            controller_str: specified controller string, e.g.
                ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner

        Returns:
            Class specified in controller_str
        Raises:
            ImportError: if module path is not valid.
            ImportError: if class name is not valid.
        """

        try:
            module_path, class_name = controller_str.rsplit(".", 1)
        except ValueError:
            raise ImportError("%s doesn't look like a module path" % controller_str)

        module = import_module(module_path)

        try:
            return getattr(module, class_name)
        except AttributeError as err:
            raise ImportError(
                'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
            ) from err
