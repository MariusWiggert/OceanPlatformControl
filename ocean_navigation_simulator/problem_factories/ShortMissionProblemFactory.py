import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from ocean_navigation_simulator.environment.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.utils import units


class ShortMissionProblemFactory(ProblemFactory):
    def __init__(
        self,
        seed: Optional[int] = None,
        config: Optional[dict] = {},
    ):
        self.config = config
        self.seed = seed
        self.random = np.random.default_rng(seed)

    def next_problem(self, skip: Optional[int] =0) -> NavigationProblem:
        if not self.has_problems_remaining():
            raise Exception("No more available Problems.")

        self.available_missions = self.available_missions[skip:]
        index = self.available_missions.pop(0)
        row = self.mission_df.iloc[index]

        return NavigationProblem.from_mission(row)

    def has_problems_remaining(self) -> bool:
        return True