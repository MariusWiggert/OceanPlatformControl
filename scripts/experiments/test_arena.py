import datetime
import numpy as np
from tqdm import tqdm

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformState
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.problem import Problem
from ocean_navigation_simulator.env.utils import units
#%%
import time
start = time.time()

arena, platform_state, observation = ArenaFactory.create(scenario_name='current_highway')
#%%
controller = NaiveToTargetController(problem=Problem(
    start_state=platform_state,
    end_region=PlatformState(
        date_time=platform_state.date_time,
        lon=units.Distance(deg=10),
        lat=units.Distance(deg=10)
    )
))
#%%

for i in tqdm(range(40)):#6 * 40)):
    action = controller.get_action(observation)
    observation = arena.step(action)
# Testing if solar caching or not-caching makes much of a difference
# For 240 steps: without caching 0.056s > with caching: 0.037.
print(arena.state_trajectory)

#arena.do_nice_plot(x_T=np.array([controller.problem.end_region.lon.deg, controller.problem.end_region.lat.deg]))

print("Total Script Time: ", time.time() - start)
