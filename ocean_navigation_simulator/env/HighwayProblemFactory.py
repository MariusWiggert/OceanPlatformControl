import datetime as dt
from typing import List, Tuple

from ocean_navigation_simulator.env.HighwayProblem import HighwayProblem
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory


class HighwayProblemFactory(ProblemFactory):
    def __init__(self, positions: List[Tuple[SpatialPoint, SpatialPoint]]):
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

    def has_problems_remaining(self) -> bool:
        return self.iter.__length_hint__() > 0
