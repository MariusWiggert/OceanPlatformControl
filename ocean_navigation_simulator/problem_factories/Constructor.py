import datetime
from importlib import import_module
from typing import Type, Union

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.ocean_observer.NoObserver import NoObserver
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.utils import units


class Constructor:
    """Class that constructs arena, problem, observer and controller objects from experiment objective, mission, arena, controller and observer configuration"""

    def __init__(
        self,
        arena_conf: dict,
        mission_conf: dict,
        objective: Union["nav", "max_seaweed"],
        ctrl_conf: dict,
        observer_conf: dict,
    ):
        """Creates the arena, problem, observer and controller objects

        Args:
            arena_conf: dictionary which specifies the arena configuration
            mission_conf: dictionary which contains the mission configuration
            objective: string which specifies which objective the experiment has i.e "nav" for navigation --> currently takes only two objectives ["nav","max_seaweed"]
            ctrl_conf: dictionary which specifies the controller configuration
            observer_conf: dictionary which specifies the observer configuration
        """
        # Init
        self.mission_conf = mission_conf
        self.objective = objective
        self.ctrl_conf = ctrl_conf
        self.observer_conf = observer_conf

        # Create problem from config
        self.problem = self.__problem_constructor()

        # Add mission seed to arena for potential forecast generation (i.e. Jonas work)
        if "seed" in self.mission_conf:
            arena_conf["seed"] = self.mission_conf["seed"]

        # Create arena from config
        self.arena = ArenaFactory.create(scenario_config=arena_conf)

        # Add platform_dict from arena to controll config
        self.ctrl_conf.update({"platform_dict": self.arena.platform.platform_dict})

        # Create controller from config
        self.controller = self.__controller_constructor()

        # Create observer from config
        self.observer = self.__observer_constructor()

        self.__problem_constructor()

    def __problem_constructor(self) -> Type[Problem]:
        """Constructs and returns problem depending on objective and mission

        Args:
            self
        Returns:
            A Problem object depending on experiment objective and mission
        """

        # TODO: Test!
        # Create PlatformState objects from mission config
        X_0 = []

        for x in self.mission_conf["x_0"]:
            X_0.append(
                PlatformState(
                    lon=units.Distance(deg=x["lon"]),
                    lat=units.Distance(deg=x["lat"]),
                    date_time=datetime.datetime.strptime(x["date_time"], "%Y-%m-%d %H:%M:%S.%f %z"),
                )
            )

        # TODO: Test!
        # Create SpatialPoint objects from mission config and save it back to mission_conf
        x_T = SpatialPoint(
            lon=units.Distance(deg=self.mission_conf["x_T"]["lon"]),
            lat=units.Distance(deg=self.mission_conf["x_T"]["lat"]),
        )

        if self.objective == "nav":
            return NavigationProblem(
                start_state=X_0[0],
                end_region=x_T,
                target_radius=self.mission_conf["target_radius"],
                timeout=self.mission_conf["timeout"],
            )

        # TODO: Adapt to new objectives i.e.:
        #
        # elif(self.objective=="max_seaweed"):
        #     # TODO: code SeaweedProblem problem class
        #     return SeaweedProblem(
        #             start_state=X_0[0],
        #             timeout=self.mission_conf["timeout"],)
        # elif(self.objective=="safety"):
        #     # TODO: code SafetyProblem class
        #     return SafetyProblem(
        #             start_state=X_0[0],
        #             end_region=x_T,
        #             target_radius=self.mission_conf["target_radius"],
        #             safety_criteria = self.mission_conf["safety_criteria"])
        # elif(self.objective=="multi-agent-nav"):
        #     # TODO: code multiAgentNavProblem problem class
        #     return multiAgentNavProblem(
        #             start_state=X_0[0],
        #             end_region=x_T,
        #             arget_radius=self.mission_conf["target_radius"],
        #             timeout=self.mission_conf["timeout"],)

    def __controller_constructor(self) -> Type[Controller]:
        """Constructs and returns the controller object depending controller configuration

        Args:
            self
        Returns:
            A Controller object depending controller configuration
        """

        # Get controller class from config
        ControllerClass = self.__import_class_from_string()

        # Return controller object
        return ControllerClass(problem=self.problem, specific_settings=self.ctrl_conf)

    def __observer_constructor(self) -> Union[Observer, NoObserver]:
        """Constructs a observer object if the observer configuration specifies a observer. If not a empty observer object is created.

        Args:
            self
        Returns:
            Either a Observer object if specified in observer configuration otherwise NoObserver object
        """
        if ("observer" not in self.observer_conf or self.observer_conf["observer"] is None):
            return NoObserver()
        else:
            return Observer(self.observer_conf["observer"])

    def __import_class_from_string(self):
        """Import a dotted module path from ctrl_conf and return the attribute/class designated by the last name in the path. Raise ImportError if the import failed.

        Args:
            self
        Returns:
            Class specified in ctrl_conf
        Raises:
            ImportError: if module path is not valid.
            ImportError: if class name is not valid.
        """

        try:
            module_path, class_name = self.ctrl_conf["ctrl_name"].rsplit(".", 1)
        except ValueError:
            raise ImportError("%s doesn't look like a module path" % self.ctrl_conf["ctrl_name"])

        module = import_module(module_path)

        try:
            return getattr(module, class_name)
        except AttributeError as err:
            raise ImportError(
                'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
            ) from err
