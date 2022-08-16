from typing import Optional, List
import numpy as np
import pandas as pd

from ocean_navigation_simulator.environment.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem


class FileMissionProblemFactory(ProblemFactory):
    def __init__(
        self,
        seed: Optional[int] = None,
        csv_file: Optional[str] = 'missions/validation/feasible.csv',
        limit: Optional[int] = None,
    ):
        self.seed = seed
        self.csv_file = csv_file
        self.limit = limit

        if limit is None:
            self.mission_df = pd.read_csv(csv_file, index_col=0)
        else:
            self.mission_df = pd.read_csv(csv_file, index_col=0).head(n=limit)
        if seed is None:
            self.available_missions = list(range(self.mission_df.shape[0]))
        else:
            self.random = np.random.default_rng(seed)
            self.available_missions = np.random.permutation(self.mission_df.shape[0]).tolist()

    def has_problems_remaining(self) -> bool:
        return len(self.available_missions) > 0

    def skips_problems(self, n) -> bool:
        self.available_missions = self.available_missions[n:]
        return self

    def get_problem_list(self, limit = None) -> [NavigationProblem]:
        if limit is not None and not len(self.available_missions) > limit:
            raise Exception(f"Only {self.available_missions} available Problems but {limit} were rquested.")

        missions = self.mission_df.iloc[self.available_missions[:limit+1] if limit is not None else self.available_missions]
        self.available_missions = self.available_missions[limit:]

        return [NavigationProblem.from_dict(row, extra_info={'index':index}) for index, row in missions.iterrows()]

    def next_problem(self) -> NavigationProblem:
        if not self.has_problems_remaining():
            raise Exception("No more available Problems.")

        index = self.available_missions.pop(0)
        row = self.mission_df.iloc[index]

        return NavigationProblem.from_row(row)