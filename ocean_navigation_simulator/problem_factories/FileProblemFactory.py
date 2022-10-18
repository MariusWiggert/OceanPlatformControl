from typing import List, Optional

import numpy as np
import pandas as pd

from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.ProblemFactory import (
    ProblemFactory,
)


class FileProblemFactory(ProblemFactory):
    """
    FileProblemFactory loads Problems from a csv file:
       - where only certain indiices can be selected
       - indices can be excluded
       - a seed can be passed for reproducability
       - a limit can be se on how many problems to return
    """

    def __init__(
        self,
        csv_file,
        indices: Optional[List] = None,
        limit: Optional[int] = None,
        exclude: Optional[List] = None,
        seed: Optional[int] = None,
    ):
        self.csv_file = csv_file
        self.indices = indices
        self.limit = limit
        self.exclude = exclude
        self.seed = seed

        self.problems_df = pd.read_csv(csv_file, index_col=0)

        self.indexes_available = self.get_indices()

    def has_problems_remaining(self) -> bool:
        return len(self.indexes_available) > 0

    def get_problem_list(self, n=None) -> [NavigationProblem]:
        if n is None:
            return [self.next_problem() for _ in range(len(self.indexes_available))]
        else:
            return [self.next_problem() for _ in range(min(n, len(self.indexes_available)))]

    def next_problem(self) -> NavigationProblem:
        if not self.has_problems_remaining():
            if self.limit is None:
                self.indexes_available = self.get_indices()
            else:
                raise Exception("No more available Problems.")

        index = self.indexes_available.pop(0)
        row = self.problems_df.iloc[index]

        return NavigationProblem.from_pandas_row(row)

    def get_indices(self):
        if self.indices is None:
            available = list(range(self.problems_df.shape[0]))

            if self.exclude is not None:
                available = [i for i in available if i not in self.exclude]

            if self.limit is not None:
                available = available[: self.limit]

            if self.seed is not None:
                random = np.random.default_rng(self.seed)
                available = random.permutation(available).tolist()

            return available
        else:
            return self.indices

    @staticmethod
    def split_indices(csv_file, split, leave=0, seed=2022):

        random = np.random.default_rng(seed)
        problems_df = pd.read_csv(csv_file, index_col=0)
        all_indices = random.permutation(problems_df.shape[0] - leave)
        return [i.tolist() for i in np.array_split(all_indices, split)]
