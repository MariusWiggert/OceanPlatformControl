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

        if limit is not None:
            self.problems_df = pd.read_csv(csv_file)
        else:
            self.problems_df = pd.read_csv(csv_file).head(n=limit)

        if seed is not None:
            self.random = np.random.default_rng(seed)
            self.indexes_available = self.random.permutation(self.problems_df.shape[0]).tolist()
        else:
            self.indexes_available = list(range(self.problems_df.shape[0]))

    def has_problems_remaining(self) -> bool:
        return len(self.indexes_available) > 0

    def get_problem_list(self, n = None) -> [NavigationProblem]:
        return [self.next_problem() for _ in range(min(n, len(self.indexes_available)))]

    def next_problem(self) -> NavigationProblem:
        if not self.has_problems_remaining():
            if self.limit is not None:
                raise Exception("No more available Problems.")
            else:
                self.indexes_available = self.random.permutation(self.problems_df.shape[0]).tolist()

        index = self.indexes_available.pop(0)
        row = self.problems_df.iloc[index]

        return NavigationProblem.from_dict(row)