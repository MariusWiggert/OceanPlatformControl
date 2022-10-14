import yaml
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog

from ocean_navigation_simulator.reinforcement_learning.OceanApexDQN import OceanApexDQN
from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv
from ocean_navigation_simulator.reinforcement_learning.OceanTorchModel import OceanTorchModel

config = yaml.load(open(f'config/reinforcement_learning/training/experiment_basic.yaml'), Loader=yaml.FullLoader)


env = OceanEnv(
    config=config['environment'],
    feature_constructor_config=config['feature_constructor'],
    reward_function_config=config['reward_function'],
    verbose=1,
)
print('action_space', env.action_space)
print('observation_space', env.observation_space)
print('reward_range', env.reward_range)
print('')


obs = env.reset()

print(obs)

policy_class = OceanApexDQN.get_default_policy_class(None, config['algorithm'])
print('policy_class', policy_class)
ModelCatalog.register_custom_model("OceanTorchModel", OceanTorchModel)
policy = policy_class(env.observation_space, env.action_space, config['algorithm'])

policy.compute_single_action(obs=obs)