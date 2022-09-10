from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.FeatureConstructor import FeatureConstructor
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.environment.Platform import PlatformAction


class RLControllerFromAgent(Controller):
    """
    RL-based Controller using
    """

    def __init__(
        self,
        problem: Problem,
        agent,
        feature_constructor: FeatureConstructor,
    ):
        """
        StraightLineController constructor
        Args:
            problem: the Problem the controller will run on
        """
        super().__init__(problem)

        self.agent = agent
        self.feature_constructor = feature_constructor

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """
        Return action that goes in the direction of the target with full power.
        Args:
            observation: observation returned by the simulator
        Returns:
            SimulatorAction dataclass
        """
        obs = self.feature_constructor.get_features_from_state(obs=observation, problem=self.problem)
        action = self.agent.compute_action(observation=obs, explore=False)

        # go towards the center of the target with full power
        return PlatformAction(magnitude=1, direction=action[0])