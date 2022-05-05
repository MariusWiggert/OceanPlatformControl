import numpy as np

import gym_envs
import gym
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.utils import check_env

from Arena import ArenaObservation
from PlatformState import PlatformState
from controllers.NaiveToTarget import NaiveToTargetController
from data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from gym_envs.envs import PlatformEnv

# trainer = PPOTrainer(env="CartPole-v0", config={"train_batch_size": 4000})
# for i in range(3):
#     print(trainer.train())

env = gym.make("DoubleGyre-v0")

controller = NaiveToTargetController(problem=env.problem)

obs = env.reset()
for i in range(200):
    action = controller.get_action(ArenaObservation(
        platform_state=PlatformState.from_numpy(np.array([obs[0], obs[1], obs[2], 0, 0])),
        true_current_at_state=OceanCurrentVector.from_numpy(np.array([obs[3], obs[4]])),
        forecasted_current_at_state=None
    ))
    obs, reward, done, _ = env.step(np.array([action.magnitude, action.direction]))
    print(obs, reward, done)




#
# # env = gym.make("DoubleGyre-v0")
# # print("check_env result: ", check_env(env))
# #
# # print(env.spec.max_episode_steps)
#
# # this line registers the environment for RLLib
# register_env("DoubleGyre-v0", lambda config: PlatformEnv())
#
# config = {
#     # Environment (RLlib understands openAI gym registered strings).
#     "env": "DoubleGyre-v0",
#     # Use 1 environment workers (aka "rollout workers") that parallelly
#     # collect samples from their own environment clone(s).
#     "num_workers": 1,
#     # Change this to "framework: torch", if you are using PyTorch.
#     # Also, use "framework: tf2" for tf2.x eager execution.
#     "framework": "torch",
#     # Tweak the default model provided automatically by RLlib,
#     # given the environment's observation- and action spaces.
#     "model": {
#         "fcnet_hiddens": [64, 64],
#         "fcnet_activation": "relu",
#     },
#     # Set up a separate evaluation worker set for the
#     # `trainer.evaluate()` call after training (see below).
#     "evaluation_num_workers": 1,
#     # Only for evaluation runs, render the env.
#     "evaluation_config": {
#         "render_env": False,
#     },
# }
#
# # Create our RLlib Trainer.
# trainer = PPOTrainer(config=config)
#
# print("starting training...")
# for i in range(3):
#     print(trainer.train())
#     print("{}th training done".format(i))
#
# trainer.evaluate()
