import datetime
import numpy as np
from tqdm import tqdm

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformState
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.utils import units
#%%
import time
start = time.time()


#arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='current_highway')
#arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='double_gyre')
arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='gulf_of_mexico')

#%%
controller = NaiveToTargetController(problem=Problem(
    start_state=platform_state,
    end_region=end_region
))
#%%

for i in tqdm(range(500)):#6 * 40)):
    action = controller.get_action(observation)
    observation = arena.step(action)
# Testing if solar caching or not-caching makes much of a difference
# For 240 steps: without caching 0.056s > with caching: 0.037.

arena.quick_plot(end_region=end_region)

#arena.plot_spatial(end_region=end_region, margin=2)
#arena.plot_spatial(end_region=end_region, margin=2, show_control=False, background=None)
#arena.plot_battery()
#arena.plot_seaweed(end_region=end_region, margin=2)
#arena.plot_control(end_region=end_region, margin=2)

print("Total Script Time: ", time.time() - start)