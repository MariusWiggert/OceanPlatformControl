import json

import ray
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQN
from ray.rllib.models import ModelCatalog

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.Arena import ArenaObservation, Arena
from ocean_navigation_simulator.environment.FeatureConstructor import FeatureConstructor
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.reinforcement_learning.OceanDenseTFModel import OceanDenseTFModel
from ocean_navigation_simulator.reinforcement_learning.OceanDenseTorchModel import OceanDenseTorchModel
from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.reinforcement_learning.OceanEnvFactory import OceanEnvFactory
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import OceanFeatureConstructor
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils


class RLController(Controller):
    """
    RL-based Controller using a pre-traine rllib trainer.
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

        Utils.ensure_storage_connection()
        with open(f'{config["experiment"]}config/config.json') as f:
            self.e_config = json.load(f)

        if self.e_config['algorithm_name'] == 'apex-dqn':
            trainer_class = ApexDQN

        self.e_config['algorithm']['num_workers'] = 0
        self.e_config['algorithm']['num_gpus'] = 0
        self.e_config['algorithm']['disable_env_checking'] = True
        self.e_config['algorithm']['optimizer']['num_replay_buffer_shards'] = 1
        self.e_config['algorithm']['log_level'] = 'ERROR'

        ray.tune.registry.register_env("OceanEnv", OceanEnvFactory(
            config=self.e_config['environment'],
            feature_constructor_config=self.e_config['feature_constructor'],
            reward_function_config=self.e_config['reward_function'],
            folders=self.e_config['folders'],
            empty_env=True,
            verbose=self.verbose-1,
        ))

        if self.e_config['algorithm']['model'].get('custom_model', '') == 'OceanDenseTFModel':
            ModelCatalog.register_custom_model("OceanDenseTFModel", OceanDenseTFModel)
        elif self.e_config['algorithm']['model'].get('custom_model', '') == 'OceanDenseTorchModel':
            ModelCatalog.register_custom_model("OceanDenseTorchModel", OceanDenseTorchModel)

        self.trainer = trainer_class(config=self.e_config['algorithm'])
        self.trainer.restore(f'{config["experiment"]}checkpoints/checkpoint_{config["checkpoint"]:06d}/')

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
        action = self.trainer.compute_single_action(observation=obs, explore=False)

        # go towards the center of the target with full power
        return PlatformAction(magnitude=1, direction=action)