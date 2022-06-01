import datetime as dt
from typing import List, Tuple

from ocean_navigation_simulator.problem_factories.HighwayProblem import HighwayProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.problem_factories.ProblemFactory import ProblemFactory


class HighwayProblemFactory(ProblemFactory):
    """ Problem factory class that creates Highway Problems gor a given list of pair of points
    """

    def __init__(self, positions: List[Tuple[SpatialPoint, SpatialPoint]]):
        self.iter = iter(positions)

    def next_problem(self) -> HighwayProblem:
        """ Get the next highway problem

        Returns:
            the next problem
        """
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
        """ tells us if the factory can still create problems

        Returns:
            True iff the factory can still create problems
        """
        return self.iter.__length_hint__() > 0
