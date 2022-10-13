import json
from ray.rllib.policy.policy import Policy

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.Arena import ArenaObservation, Arena
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import OceanFeatureConstructor
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils


class RLController(Controller):
    """
    RL-based Controller using a pre-traine rllib policy.
    """
    def __init__(
        self,
        config: dict,
        problem: NavigationProblem,
        arena: Arena,
        verbose: int = 0,
    ):
        super().__init__(problem, verbose)
        self.config = config
        self.arena = arena

        # Step 1: Recover Configuration
        Utils.ensure_storage_connection()
        with open(f'{config["experiment"]}config/config.json') as f:
            self.e_config = json.load(f)

        # Step 2: Modify Configuration
        self.e_config['algorithm']['num_workers'] = 0
        self.e_config['algorithm']['num_gpus'] = 0
        self.e_config['algorithm']['disable_env_checking'] = True
        self.e_config['algorithm']['optimizer']['num_replay_buffer_shards'] = 1
        self.e_config['algorithm']['log_level'] = 'ERROR'

        # Step 3: Create Trainer
        self.policy = Policy.from_checkpoint(
            f'{config["experiment"]}checkpoints/checkpoint_{config["checkpoint"]:06d}/policies/default_policy'
        )

        # Step 4: Create Feature Constructor
        self.hindcast_planner = HJReach2DPlanner.from_plan(
            folder=f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/hindcast_planner/',
            problem=self.problem,
            verbose=self.verbose - 1,
        )
        self.forecast_planner = HJReach2DPlanner.from_plan(
            folder=f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/forecast_planner_idx_0/',
            problem=self.problem,
            specific_settings={
                'load_plan': True,
                'planner_path': f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/',
            },
            verbose=self.verbose - 1,
        )
        self.feature_constructor = OceanFeatureConstructor(
            forecast_planner=self.forecast_planner,
            hindcast_planner=self.hindcast_planner,
            config=self.e_config['feature_constructor'],
            verbose=self.verbose - 1
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