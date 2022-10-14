import ray
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger

from ocean_navigation_simulator.reinforcement_learning.OceanApexDQN import OceanApexDQN
from ocean_navigation_simulator.reinforcement_learning.OceanDenseTFModel import OceanDenseTFModel
from ocean_navigation_simulator.reinforcement_learning.OceanTorchModel import OceanTorchModel
from ocean_navigation_simulator.reinforcement_learning.OceanEnvFactory import OceanEnvFactory


class TrainerFactory:
	served = 0

	@staticmethod
	def create(config, checkpoint=None, logger_path=None, verbose=0):
		# Step 1: Register Env
		ray.tune.registry.register_env("OceanEnv", OceanEnvFactory(
			config=config['environment'],
			feature_constructor_config=config['feature_constructor'],
			reward_function_config=config['reward_function'],
			num_workers=config['algorithm']['num_workers'],
			result_root=f"{config['folders']['experiment']}workers",
			verbose=verbose,
		))

		# Step 2: Register Model
		if config['algorithm']['model'].get('custom_model', '') == 'OceanDenseTFModel':
			ModelCatalog.register_custom_model("OceanDenseTFModel", OceanDenseTFModel)
		elif config['algorithm']['model'].get('custom_model', '') == 'OceanTorchModel':
			ModelCatalog.register_custom_model("OceanTorchModel", OceanTorchModel)

		# Step 3: Select Class
		if config['algorithm_name'] == 'apex-dqn':
			trainer_class = OceanApexDQN
		else:
			raise ValueError(f"Algorithm '{config['algorithm_name']}' not implemented.")

		# Step 4: Create Trainer
		if logger_path is not None:
			trainer = trainer_class(config=config['algorithm'], logger_creator=lambda config: UnifiedLogger(config, logger_path, loggers=None))
		else:
			trainer = trainer_class(config=config['algorithm'])

		# Step 4: Restore Checkpoint
		if checkpoint is not None:
			trainer.restore(checkpoint)

		return trainer