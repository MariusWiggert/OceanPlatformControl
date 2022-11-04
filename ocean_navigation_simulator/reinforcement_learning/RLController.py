import json

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

from ocean_navigation_simulator.controllers.Controller import Controller

# from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
#     HJReach2DPlanner,
# )
from ocean_navigation_simulator.environment.Arena import (
    Arena,
    ArenaObservation,
)

# from ocean_navigation_simulator.environment.NavigationProblem import (
#     NavigationProblem,
# )
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.reinforcement_learning.env.OceanFeatureConstructor import (
    OceanFeatureConstructor,
)
from ocean_navigation_simulator.reinforcement_learning.missions.CachedNavigationProblem import (
    CachedNavigationProblem,
)
from ocean_navigation_simulator.reinforcement_learning.OceanTorchModel import (
    OceanTorchModel,
)
from ocean_navigation_simulator.utils import cluster_utils

# Note: Needs access to arena right now because HJPlanner needs it to check for changing cached TTR map.
# When we iterate on it, might be easier to have a HJPlanner light that just indexes into the correct cached TTR map.


class RLController(Controller):
    """
    RL-based Controller using a pre-trained rllib policy. It needs access to arena to modify
    the observation with the data source needed by HJ Planner.
    """

    def __init__(
        self,
        config: dict,
        problem: CachedNavigationProblem,
        arena: Arena,
    ):
        super().__init__(problem)
        self.config = config
        self.arena = arena

        # Step 1: Recover Configuration
        cluster_utils.ensure_storage_connection()
        with open(f'{config["controller"]["experiment"]}config/config.json') as f:
            self.experiment_config = json.load(f)

        # Step 2: Create Policy
        ModelCatalog.register_custom_model("OceanTorchModel", OceanTorchModel)

        self.policy = Policy.from_checkpoint(
            f'{config["controller"]["experiment"]}checkpoints/checkpoint_{config["controller"]["checkpoint"]:06d}/policies/default_policy'
        )

        # Step 3: Create Feature Constructor
        self.hindcast_planner = problem.get_cached_hindcast_planner(
            self.config["missions"]["folder"]
        )
        self.forecast_planner = problem.get_cached_forecast_planner(
            self.config["missions"]["folder"]
        )

        self.feature_constructor = OceanFeatureConstructor(
            forecast_planner=self.forecast_planner,
            hindcast_planner=self.hindcast_planner,
            config=self.experiment_config["feature_constructor"],
        )

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """
        Return action that goes in the direction of the target with full power.
        Args:
            observation: observation returned by the simulator
        Returns:
            SimulatorAction dataclass
        """
        self.forecast_planner.replan_if_necessary(observation)
        self.hindcast_planner.replan_if_necessary(
            observation.replace_datasource(self.arena.ocean_field.hindcast_data_source)
        )
        obs = self.feature_constructor.get_features_from_state(
            obs=observation,
            problem=self.problem,
        )
        action, _, _ = self.policy.compute_single_action(obs=obs, explore=False)
        direction = 2 * np.pi * action / self.experiment_config["environment"]["actions"]

        if self.experiment_config["environment"]["fake"] == "residual":
            direction = self.forecast_planner.get_action(observation=self.prev_obs).direction

        # go towards the center of the target with full power
        return PlatformAction(magnitude=1, direction=direction)
