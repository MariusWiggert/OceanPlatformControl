from typing import Optional

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner

class HJController(Controller):
    def __init__(self, problem: Problem, platform_dict: dict, verbose: Optional[bool] = False):
        specific_settings = {
            'replan_on_new_fmrc': True,
            'replan_every_X_seconds': None,
            'direction': 'multi-time-reach-back',
            'n_time_vector': 199,  # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
            'deg_around_xt_xT_box': 4.0,  # area over which to run HJ_reachability
            'accuracy': 'high',
            'artificial_dissipation_scheme': 'local_local',
            'T_goal_in_seconds': 3600 * 119,
            'use_geographic_coordinate_system': True,
            'progress_bar': verbose,
            'initial_set_radii': [0.1, 0.1],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
            # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
            'grid_res': 0.04,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
            'd_max': 0.0,
            # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
            # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
            'platform_dict': platform_dict
        }
        self.planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings, verbose=verbose)

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        return self.planner.get_action(observation=observation)