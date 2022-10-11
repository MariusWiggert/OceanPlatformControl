from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv


class OceanEnvFactory:
	def __init__(self, config, feature_constructor_config, reward_function_config, folders, empty_env, verbose):
		self.config = config
		self.feature_constructor_config = feature_constructor_config
		self.reward_function_config = reward_function_config
		self.folders = folders
		self.empty_env = empty_env
		self.verbose = verbose

	def __call__(self, env_config):
        # env_config: env_config.num_workers, env_config.worker_index, env_config.vector_index, env_config.remote
		return OceanEnv(
            config=self.config,
            feature_constructor_config=self.feature_constructor_config,
            reward_function_config=self.reward_function_config,
            folders=self.folders,
            env_config=env_config,
            worker_index=env_config.worker_index,
			empty_env=self.empty_env,
            verbose=self.verbose
        )