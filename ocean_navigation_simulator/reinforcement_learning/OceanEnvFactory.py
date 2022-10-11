from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv


class OceanEnvFactory:
	def __init__(self, config, feature_constructor_config, reward_function_config, folders, verbose):
		self.config = config
		self.feature_constructor_config = feature_constructor_config
		self.reward_function_config = reward_function_config
		self.folders = folders
		self.verbose = verbose

	def __call__(self, env_config):
		return OceanEnv(
            config=self.config,
            feature_constructor_config=self.feature_constructor_config,
            reward_function_config=self.reward_function_config,
            folders=self.folders,
            worker_index=env_config.worker_index,
            env_config=env_config,
            verbose=self.verbose-1
        )