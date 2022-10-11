from typing import Optional, List
import numpy as np
import pandas as pd

from ocean_navigation_simulator.environment.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem


class FileMissionProblemFactory(ProblemFactory):
    def __init__(
        self,
        seed: Optional[int] = None,
        csv_file: Optional[str] = '/seaweed-storage/generation/gulf_of_mexico_HYCOM_hindcast/increased_planner_area/problems.csv',
        indices: Optional[List] = None,
        limit: Optional[int] = None,
    ):
        self.seed = seed
        self.csv_file = csv_file
        self.indices = indices
        self.limit = limit

        self.problems_df = pd.read_csv(csv_file, index_col=0)

        self._set_indices()

    def has_problems_remaining(self) -> bool:
        return len(self.indexes_available) > 0

    def get_problem_list(self, n = None) -> [NavigationProblem]:
        if n is None:
            return [self.next_problem() for _ in range(len(self.indexes_available))]
        else:
            return [self.next_problem() for _ in range(min(n, len(self.indexes_available)))]

    def next_problem(self) -> NavigationProblem:
        if not self.has_problems_remaining():
            if self.limit is None:
                self._set_indices()
            else:
                raise Exception("No more available Problems.")

        index = self.indexes_available.pop(0)
        row = self.problems_df.iloc[index]

        return NavigationProblem.from_pandas_row(row)

    def _set_indices(self):
        if self.indices is None:
            if self.limit is None:
                available = self.problems_df.shape[0]
            else:
                available = self.limit

            if self.seed is None:
                self.indexes_available = list(range(available))
            else:
                self.random = np.random.default_rng(self.seed)
                self.indexes_available = self.random.permutation(available).tolist()
        else:
            self.indexes_available = self.indices