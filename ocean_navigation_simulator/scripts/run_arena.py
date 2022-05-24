import datetime
import numpy as np
from tqdm import tqdm

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.Problem import Problem

#%%
import time

def run_arena(
        arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast'),
        controller = ,
        max_steps=600
):
    start = time.time()

    arena =

    #%%
    controller = NaiveToTargetController(problem=Problem(
        start_state=problem.start_state,
        end_region=problem.end_region
    ))
    #%%

    for i in tqdm(range(max_steps)):#6 * 40)):
        action = controller.get_action(observation)
        observation = arena.step(action)


    arena.quick_plot(end_region=problem.end_region)

    #arena.plot_spatial(end_region=end_region, margin=2)
    #arena.plot_spatial(end_region=end_region, margin=2, show_control=False, background=None)
    #arena.plot_battery()
    #arena.plot_seaweed(end_region=end_region, margin=2)
    #arena.plot_control(end_region=end_region, margin=2)

    print("Total Script Time: ", time.time() - start)


if __name__ == "__main__":
    run_arena()