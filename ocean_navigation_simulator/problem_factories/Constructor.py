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
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.ocean_observer.NoObserver import NoObserver
from ocean_navigation_simulator.ocean_observer.Observer import Observer


class Constructor:
    """Class that constructs arena, problem, observer and controller objects from configuration files of
    experiment objective, mission, arena, controller and observer"""

    def __init__(
        self,
        arena_conf: dict,
        mission_conf: dict,
        objective_conf: dict,
        ctrl_conf: dict,
        observer_conf: dict,
        # Relevant for downloading data
        throw_exceptions=True,
        c3=None,
        download_files=False,
        timeout_in_sec=0,
        create_arena=True,
    ):
        """Creates the arena, problem, observer and controller objects

        Args:
            arena_conf: dictionary which specifies the arena configuration
            mission_conf: dictionary which contains the mission configuration
            objective: dict which specifies which objective the experiment has under 'type' i.e 'type': "nav" for navigation
            ctrl_conf: dictionary which specifies the controller configuration
            observer_conf: dictionary which specifies the observer configuration
            c3: if running on C3, the c3 object is passed in directly, locally pass in nothing
            download_files: if the current files should be downloaded
            timeout_in_sec: this is used to determine what current files to download such that there is enough data
                            until the problem times out.
        """
        # Init
        self.mission_conf = mission_conf
        self.objective_conf = objective_conf
        self.ctrl_conf = ctrl_conf
        self.observer_conf = observer_conf

        # Create arena from config
        if create_arena:
            # Add mission seed to arena for potential forecast generation (i.e. Jonas work)
            if "seed" in self.mission_conf:
                arena_conf["seed"] = self.mission_conf["seed"]

            # get t_interval for downloading files
            point_to_check = SpatioTemporalPoint.from_dict(mission_conf["x_0"][0])
            t_interval = [
                point_to_check.date_time,
                point_to_check.date_time
                + datetime.timedelta(
                    seconds=timeout_in_sec
                    + arena_conf["casadi_cache_dict"]["time_around_x_t"]
                    + 7200
                ),
            ]
            self.arena = ArenaFactory.create(
                scenario_config=arena_conf,
                t_interval=t_interval if download_files else None,
                c3=c3,
                throw_exceptions=throw_exceptions,
            )

        # Add platform_dict from arena to controll config
        self.platform_dict = arena_conf["platform_dict"]

        # Create problem from config
        self.problem = self.__problem_constructor()

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

        # handle user error when fed in not as list
        if type(self.mission_conf["x_0"]) == dict:
            raise TypeError(
                "mission_conf[x_0] needs to be a list of state dicts!, not a dict itself."
            )

        for x in self.mission_conf["x_0"]:
            X_0.append(PlatformState.from_dict(x))

        # TODO: Test!
        # Create SpatialPoint objects from mission config and save it back to mission_conf
        x_T = SpatialPoint.from_dict(self.mission_conf["x_T"])

        if self.objective_conf["type"] == "nav":
            return NavigationProblem(
                start_state=X_0[0],
                end_region=x_T,
                target_radius=self.mission_conf["target_radius"],
                platform_dict=self.platform_dict,
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
        if "observer" not in self.observer_conf or self.observer_conf["observer"] is None:
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
