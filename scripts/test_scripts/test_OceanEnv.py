import time

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

script_start = time.time()

env = OceanEnv(verbose=10)
env.reset()
controller = NaiveController(problem=env.problem)

print(f'########## Setup Time: {time.time()-script_start:.1f}s')

step = 0
done = False
total_reward = 0

# print(f'Initial TTR in h: {env.hindcast_planner.interpolate_value_function_in_hours_at_point(observation=env.prev_obs)}')

while not done:
    action = controller.get_action(env.prev_obs)

    start = time.time()
    features, reward, done, info = env.step(action)
    # print(f'OceanEnv Step {step} ({time.time()-start:.1f}s)')
    # print(features.shape, reward, done, info)
    total_reward += reward
    # print(f'Current TTR in h at Step {step}: {env.hindcast_planner.interpolate_value_function_in_hours_at_point(observation=env.prev_obs):.2f}')
    # print(f'Total Reward (Improvement in TTR in h): {total_reward:.2f}')

    step += 1

# print(f'Final TTR in : {env.hindcast_planner.interpolate_value_function_in_hours_at_point(observation=env.prev_obs)}')
# print(f'Passed Time in h: {env.problem.passed_seconds(env.arena.platform.state) / 3600:.2f}h')

print(f'### Mean Env Step Time: {(time.time()-script_start)/200:.3f}s')
print(f'### Script Time: {time.time()-script_start:.1f}s')