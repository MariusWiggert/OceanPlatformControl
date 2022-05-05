import abc
import datetime as dt
from typing import Optional, List, Tuple

import numpy as np

from ocean_navigation_simulator.env.DoubleGyreProblem import DoubleGyreProblem
from ocean_navigation_simulator.env.HighwayProblem import HighwayProblem
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.env.utils import units


class HighwayProblemFactory(ProblemFactory):
    def __init__(self, positions: List[Tuple[SpatialPoint,SpatialPoint]]):
        self.iter = iter(positions)

    def next_problem(self) -> HighwayProblem:
        positions = next(self.iter)
        if positions is None:
            raise StopIteration()
        start, end = positions
        start_state = PlatformState(
            start.lon,
            start.lat,
            date_time=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
        )

        return HighwayProblem(
            start_state=start_state,
            end_region=end
        )

    def has_problems_remaining(self) -> int:
        return self.iter.__length_hint__() > 0