from typing import List

from ocean_navigation_simulator import Problem
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory


class NaiveProblemFactory(ProblemFactory):
    """Create a naive problem factory that simply return the problems from a given list successively
    """

    def __init__(self, problems: List[Problem]):
        """ Create the factory
        Args:
            problems: List of problems that will be generated
        """
        self.problems = problems

    def next_problem(self) -> Problem:
        """Gives us the next problem of the list

        Returns:
            The next problem
        """
        if not len(self.problems):
            raise StopIteration()
        return self.problems.pop(0)

    def has_problems_remaining(self) -> bool:
        """ Tells us if the factory can still create new problems

        Returns:
            True iff the factory can still create at least one problem
        """
        return len(self.problems) > 0