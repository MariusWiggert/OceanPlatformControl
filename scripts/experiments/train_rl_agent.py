#%%

import gym
import ray.rllib.utils
from ray.rllib.agents.ppo import PPOTrainer
import pickle

from ocean_navigation_simulator.env.PlatformEnv import PlatformEnv

gym.envs.register(
    id='DoubleGyre-v0',
    entry_point='ocean_navigation_simulator.env.PlatformEnv:PlatformEnv',
    kwargs={
        'seed': 2022,
        'env_steps_per_arena_steps': 10,
    },
    max_episode_steps=1000,
)
env = gym.make('DoubleGyre-v0')
ray.tune.registry.register_env("DoubleGyre-v0", lambda config: PlatformEnv())

#%%

########## TEST ENV ##########
# controller = NaiveToTargetController(problem=env.problem)
#
# obs = env.reset()
# for i in range(200):
#     action = controller.get_action(ArenaObservation(
#         platform_state=PlatformState.from_numpy(np.array([obs[0], obs[1], obs[2], 0, 0])),
#         true_current_at_state=OceanCurrentVector.from_numpy(np.array([obs[3], obs[4]])),
#         forecast_data_source=None
#     ))
#     obs, reward, done, _ = env.step(np.array([action.magnitude, action.direction]))
#     print(obs, reward, done)

#%%
########## TRAIN FIRST RL ##########

config = {
    # === Environment Settings ===
    # Environment (RLlib understands openAI gym registered strings).
    "env": "DoubleGyre-v0",

    # === Settings for Rollout Worker processes ===
    # Use 1 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 1,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # 'disable_env_checking': False,
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },

    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },

    # === Settings for the Trainer process ===

}
#
# Create our RLlib Trainer.
agent = PPOTrainer(config=config)
pickle.dump(config, open(f'ocean_navigation_simulator/models/simplified_double_gyre/config.p', "wb"))
#%%
print("starting training...")

for i in range(200):
    print(agent.train())
    agent.save(f'ocean_navigation_simulator/models/simplified_double_gyre/')
    print(f'{i}th training done')

print('training done!')
#%%
agent.evaluate()
#%%





