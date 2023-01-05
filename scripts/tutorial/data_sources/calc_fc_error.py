# This File will calculate the FC Error for a specific problem and FC/HC Setting
# Step 1: Create the arena and the problem object!
import logging

import yaml

from ocean_navigation_simulator.problem_factories.Constructor import (
    Constructor,
)
from ocean_navigation_simulator.utils.misc import set_arena_loggers

set_arena_loggers(logging.DEBUG)

x_0 = {
    "lon": -82.5,
    "lat": 23.7,
    "date_time": "2022-09-10 23:12:00.004573 +0000",
}

x_T = {"lon": -80.3, "lat": 24.6}

mission_config = {
    "x_0": [x_0],
    "x_T": x_T,
    "target_radius": 0.1,
    "seed": 12344093,
}

with open("config/arena/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml") as f:
    arena_config = yaml.load(f, Loader=yaml.FullLoader)
ctrl_config = {
    "ctrl_name": "ocean_navigation_simulator.controllers.NaiveController.NaiveController"
}
objective_conf = {"type": "nav"}

# Step 0: Create Constructor object which contains arena, problem, controller and observer
constructor = Constructor(
    arena_conf=arena_config,
    mission_conf=mission_config,
    objective_conf=objective_conf,
    ctrl_conf=ctrl_config,
    observer_conf={"observer": None},
    download_files=True,
)

# Step 1.1 Retrieve problem
problem = constructor.problem

# Step 1.2: Retrieve arena
arena = constructor.arena
observation = arena.reset(platform_state=problem.start_state)

feasibility_dict = {"deg_around_xt_xT_box": 1, "T_goal_in_seconds": 3600 * 24 * 2}
#%% Calculate the FC Errors
from ocean_navigation_simulator.utils.calc_fmrc_error import calc_fmrc_errors

error_dict = calc_fmrc_errors(
    problem=problem,
    arena=arena,
    t_horizon_in_h=100,
    deg_around_x0_xT_box=1,
    T_goal_in_seconds=3600 * 24 * 2,
)
