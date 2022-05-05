from gym.envs.registration import register

register(id='DoubleGyre-v0', entry_point='gym_envs.envs:PlatformEnv', max_episode_steps=200)
