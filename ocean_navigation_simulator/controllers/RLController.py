import json

from ray.rllib.policy.policy import Policy

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.environment.Arena import (
    Arena,
    ArenaObservation,
)
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import (
    OceanFeatureConstructor,
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
        problem: NavigationProblem,
        arena: Arena,
    ):
        super().__init__(problem)
        self.config = config
        self.arena = arena

        # Step 1: Recover Configuration
        cluster_utils.ensure_storage_connection()
        with open(f'{config["experiment"]}config/config.json') as f:
            self.experiment_config = json.load(f)

        # Step 2: Create Policy
        self.policy = Policy.from_checkpoint(
            f'{config["experiment"]}checkpoints/checkpoint_{config["checkpoint"]:06d}/policies/default_policy'
        )

        # Step 3: Create Feature Constructor
        self.hindcast_planner = HJReach2DPlanner.from_saved_planner_state(
            folder=f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/hindcast_planner/',
            problem=self.problem,
        )
        self.forecast_planner = HJReach2DPlanner.from_saved_planner_state(
            folder=f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/forecast_planner_idx_0/',
            problem=self.problem,
            specific_settings={
                "load_plan": True,
                "planner_path": f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/',
            },
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
        obs = self.feature_constructor.get_features_from_state(
            fc_obs=observation,
            hc_obs=observation.replace_datasource(self.arena.ocean_field.hindcast_data_source),
            problem=self.problem,
        )
        action, _, _ = self.policy.compute_single_action(obs=obs, explore=False)

        # go towards the center of the target with full power
        return PlatformAction(magnitude=1, direction=action)
