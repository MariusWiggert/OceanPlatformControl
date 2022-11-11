
from datetime import datetime, timezone

from typing import Dict, List, Optional

# Note: if you develop on hj_reachability repo and this library simultaneously, add the local version with this line
# sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) + 'hj_reachability_c3'])

import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr
import dataclasses
import multiprocessing as mp
import concurrent.futures

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformAction, PlatformActionSet
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.utils import units

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)


class MultiAgentPlanner(Controller):

    def __init__(self, problem: NavigationProblem,  multi_agent_setting: Dict, \
                specific_settings: Optional[Dict] = None):
        super().__init__(problem)

        if multi_agent_setting["planner"] == "hj_planner":
            self.individual_problems_list = [dataclasses.replace(problem, start_state= x) for x in problem.start_state]
            self.planners = [HJReach2DPlanner(problem,specific_settings=specific_settings) for problem in self.individual_problems_list]

        # initialize vectors for open_loop control
        self.times, self.x_traj, self.contr_seq = None, None, None

         # saving the planned trajectories for inspection purposes
        self.planned_trajs = []

    def get_action(self, observation: List[ArenaObservation]) -> PlatformActionSet:
        action_list = []
        action = self.planners[0].get_action(observation[0])
        for k in range(len(observation)):
            action_list.append(self.planners[k].get_action(observation[k]))
            #action_list.append(action)
        return PlatformActionSet(action_list)
    

    
    # def get_action(self, observation: List[ArenaObservation]) -> PlatformActionSet:
    #     action_list = []
    #     nb_observations = len(observation)
    #     obs_planner_id = [(obs,planner_id) for obs, planner_id in zip(observation, list(range(nb_observations)))]
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    #         for obs_planner_id, result in zip(obs_planner_id, executor.map(self._get_action_multiprocessing, obs_planner_id)):
    #             action_list.append(result)
    #         # from_executor = {executor.submit(self._get_action_multiprocessing, obs, 0): for obs, planner_id in zip(observation, list(range(nb_observations)))}
    #         # for future in concurrent.futures.as_completed(from_executor):
    #         #     action = from_executor.result()
    #     return action_list

    # # def get_action(self, observation: List[ArenaObservation]) -> PlatformActionSet:
    # #     nb_observations = len(observation)
    # #     p = mp.Pool(processes=nb_observations)
    # #     actions = p.starmap(self._get_action_multiprocessing, [(obs, planner_id) for obs, planner_id in zip(observation, list(range(nb_observations)))])
    # #     actions = p.map(self._get_action_multiprocessing, [1,2])
    # #     return actions
    # #     action_list = []
    # #             for k in range(len(observation)):

    # def _get_action_multiprocessing(self, obs, planner_id): # -> PlatformAction:
    #     return self.planners[planner_id].get_action(obs)

    