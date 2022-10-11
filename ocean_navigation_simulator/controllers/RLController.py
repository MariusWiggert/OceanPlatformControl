import json

import ray
from ray.rllib.agents.dqn.apex import ApexTrainer
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
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import OceanFeatureConstructor
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils


class RLController(Controller):
    """
    RL-based Controller using a pre-traine rllib agent.
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

        self.problem = problem
        Utils.ensure_storage_connection()
        with open(f'{config["controller"]["experiment"]}config/config.json') as f:
            self.config = json.load(f)

        if self.config['algorithm_name'] == 'apex-dqn':
            agent_class = ApexTrainer

        self.config['algorithm']['num_workers'] = 1
        self.config['algorithm']['num_gpus'] = 0

        ray.tune.registry.register_env("OceanEnv", lambda env_config: OceanEnv(
            config=self.config['environment'],
            feature_constructor_config=self.config['feature_constructor'],
            reward_function_config=self.config['reward_function'],
            folders=self.config['folders'],
            worker_index=env_config.worker_index,
            env_config=env_config,
            verbose=self.verbose-1
        ))

        if self.config['algorithm']['model'].get('custom_model', '') == 'OceanDenseTFModel':
            ModelCatalog.register_custom_model("OceanDenseTFModel", OceanDenseTFModel)
        elif self.config['algorithm']['model'].get('custom_model', '') == 'OceanDenseTorchModel':
            ModelCatalog.register_custom_model("OceanDenseTorchModel", OceanDenseTorchModel)

        self.agent = agent_class(config=self.config['algorithm'])
        self.agent.restore(f'{config["controller"]["experiment"]}{config["controller"]["checkpoint"]}')

        self.hindcast_planner = HJReach2DPlanner.from_plan(
            folder=f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/hindcast_planner/',
            problem=self.problem,
            specific_settings={
                'save_after_planning': False,
            },
            verbose=self.verbose - 1,
        )
        self.forecast_planner = HJReach2DPlanner.from_plan(
            folder=f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/forecast_planner_idx_0/',
            problem=self.problem,
            specific_settings={
                'load_plan': True,
                'planner_path': f'{self.config["missions"]["folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/',
                'save_after_planning': False,
            },
            verbose=self.verbose - 1,
        )

        self.feature_constructor = OceanFeatureConstructor(
            forecast_planner=self.forecast_planner,
            hindcast_planner=self.hindcast_planner,
            config=self.config['feature_constructor'],
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
        action = self.agent.compute_action(observation=obs, explore=False)

        # go towards the center of the target with full power
        return PlatformAction(magnitude=1, direction=action[0])