import os
from typing import Optional

import numpy as np
import pandas as pd

from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.ProblemFactory import (
    ProblemFactory,
)
from ocean_navigation_simulator.reinforcement_learning.missions.CachedNavigationProblem import (
    CachedNavigationProblem,
)


class FileProblemFactory(ProblemFactory):
    """
    FileProblemFactory loads Problems from a csv file:
       - where only certain indices can be selected
       - indices can be excluded
       - a seed can be passed for reproducability
       - a limit can be se on how many problems to return
    """

    def __init__(
        self,
        csv_file,
        filter: Optional[dict] = {},
        seed: Optional[int] = None,
    ):
        self.csv_file = csv_file
        self.filter = filter
        self.seed = seed

        self.base_path = os.path.dirname(csv_file)

        self.df = pd.read_csv(csv_file, index_col=0).reset_index()
        self.indices = FileProblemFactory.filter(self.df, self.filter, self.seed)

    def has_problems_remaining(self) -> bool:
        return len(self.indices) > 0

    def get_problem_list(self, n=None) -> [NavigationProblem]:
        if n is None:
            return [self.next_problem() for _ in range(len(self.indices))]
        else:
            return [self.next_problem() for _ in range(min(n, len(self.indices)))]

    def next_problem(self) -> CachedNavigationProblem:
        if not self.has_problems_remaining():
            if self.filter.get("finite", False):
                raise Exception("No more available Problems.")
            else:
                self.indices = FileProblemFactory.filter(self.df, self.filter, self.seed)

        row = self.df.iloc[self.indices.pop(0)]

        return CachedNavigationProblem.from_pandas_row(row, self.base_path)

    @staticmethod
    def filter(df, filter, seed=None):
        if filter.get("indices", False):
            indices = filter["indices"]
        else:
            if filter.get("no_random", False) and "random" in df:
                df = df[~df["random"]]

            if filter.get("index_start", False):
                df = df[filter["index"] > filter["index_start"]]

            if filter.get("index_stop", False):
                df = df[filter["index"] < filter["index_stop"]]

            if filter.get("starts_per_target", False):
                df = df.groupby("batch").head(filter["starts_per_target"])

            if filter.get("exclude", False):
                df = df.drop(filter["exclude"])

            if filter.get("start", False):
                df = df[filter["start"] :]

            if filter.get("stop", False):
                df = df[: filter["stop"]]

            if filter.get("limit", False):
                df = df[: filter["limit"]]

            indices = df.index.to_list()

        if seed is not None:
            random = np.random.default_rng(seed)
            indices = random.permutation(indices).tolist()

        return indices

    @staticmethod
    def split_indices(csv_file, filter, split, seed=2022):
        df = pd.read_csv(csv_file, index_col=0).reset_index()

        indices = FileProblemFactory.filter(df, filter, seed)

        return [i.tolist() for i in np.array_split(np.array(indices), split)]
