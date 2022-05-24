from typing import List

from ocean_navigation_simulator import Problem
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory


class NaiveProblemFactory(ProblemFactory):
    def __init__(self, problems: List[Problem]):
        self.problems = problems

    def next_problem(self) -> Problem:
        if not len(self.problems):
            raise StopIteration()
        return self.problems.pop(0)

    def has_problems_remaining(self) -> bool:
        return len(self.problems) > 0
