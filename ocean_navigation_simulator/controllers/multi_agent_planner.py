import concurrent.futures
import dataclasses
import multiprocessing as mp
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import (
    PlatformAction,
    PlatformActionSet,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    PlatformStateSet,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.utils import units

# Note: if you develop on hj_reachability repo and this library simultaneously, add the local version with this line
# sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) + 'hj_reachability_c3'])


class MultiAgentPlanner(Controller):
    def __init__(
        self,
        problem: NavigationProblem,
        multi_agent_setting: Dict,
        specific_settings: Optional[Dict] = None,
    ):
        super().__init__(problem)

        if multi_agent_setting["planner"] == "hj_planner":
            if not type(problem.start_state) is PlatformStateSet:
                raise Exception(
                    "Mutli-Agent Planner does not support single platform control for now"
                )
            else:
                self.individual_problems_list = [
                    dataclasses.replace(problem, start_state=x) for x in problem.start_state
                ]
                self.planners = [
                    HJReach2DPlanner(problem, specific_settings=specific_settings)
                    for problem in self.individual_problems_list
                ]

        # initialize vectors for open_loop control
        self.times, self.x_traj, self.contr_seq = None, None, None

        # saving the planned trajectories for inspection purposes
        self.planned_trajs = []

    def get_action(self, observation: List[ArenaObservation]) -> PlatformActionSet:
        action_list = []
        for k in range(len(observation)):
            single_obs = dataclasses.replace(
                observation,
                platform_state=observation.platform_state[k],
                true_current_at_state=observation.true_current_at_state[k],
            )
            action_list.append(self.planners[k].get_action(single_obs))

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
