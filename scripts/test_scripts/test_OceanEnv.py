import time
import numpy as np

from ocean_navigation_simulator.controllers.NaiveToTargetController import NaiveToTargetController
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.PlatformState import PlatformState
from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv

env = OceanEnv()
controller = NaiveToTargetController(problem=env.problem)

obs = env.reset()
start = time.time()
for i in range(200):
    action = controller.get_action(ArenaObservation(
        platform_state=PlatformState.from_numpy(np.array([obs[0], obs[1], obs[2], 0, 0])),
        true_current_at_state=OceanCurrentVector.from_numpy(np.array([0, 0])),
        forecast_data_source=None
    ))
    inter = time.time()
    obs, reward, done, _ = env.step(np.array([action.magnitude, action.direction]))
    print(f'env step took {time.time()-inter}s')
    print(obs, reward, done)
print(f'mean env step time {(time.time()-start)/200}s')