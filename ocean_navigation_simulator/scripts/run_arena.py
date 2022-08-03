import datetime
import numpy as np
from tqdm import tqdm

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.controllers.NaiveToTargetController import NaiveToTargetController

#%%
import time

def run_arena(
    arena: Arena,
    problem: Problem,
    controller: Controller,
    steps: int = 600,
):
    start = time.time()

    #%%
    controller = NaiveToTargetController(problem=Problem(
        start_state=problem.start_state,
        end_region=problem.end_region
    ))
    #%%



    print("Total Script Time: ", time.time() - start)

if __name__ == "__main__":
    run_arena()