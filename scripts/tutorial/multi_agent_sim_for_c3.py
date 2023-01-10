# This script shows how the closed-loop simulation is run in the C3 cloud, using only configs and the constructor
# %% imports
import logging

from ocean_navigation_simulator.problem_factories.Constructor import (
    Constructor,
)

logging.basicConfig(level=logging.INFO)

NoObserver = {"observer": None}

# Controller Configs
HJMultiTimeConfig = {
    "replan_every_X_seconds": None,
    "replan_on_new_fmrc": True,
    "T_goal_in_seconds": 259200,  # 3d, 43200,     # 12h
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "ctrl_name": "ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner",
    "d_max": 0.0,
    "deg_around_xt_xT_box": 1.0,
    "direction": "multi-time-reach-back",
    "grid_res": 0.02,
    "n_time_vector": 200,
    "progress_bar": True,
    "use_geographic_coordinate_system": True,
}
StraightLineConfig = {
    "ctrl_name": "ocean_navigation_simulator.controllers.NaiveController.NaiveController"
}
flocking_config = {
    "unit": "km",
    "interaction_range": 9, # km
    "grad_clip_range": 0.1, # km
}

MultiAgentConfig = {
    "high_level_ctrl": "hj_naive",
    "unit": "km",
    "communication_thrsld": 9,
    "hj_specific_settings": HJMultiTimeConfig,
}