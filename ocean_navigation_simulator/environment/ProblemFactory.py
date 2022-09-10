import abc
from typing import List

from ocean_navigation_simulator.environment import NavigationProblem

"""
Problem Factories generate Problems. They can either be completely random and infinite
or they use a pre-generated finite list. For reproducibility a seed can be fed at generation.
"""

class ProblemFactory(abc.ABC):
    """
    Interface for a problem generator.
    """

    @abc.abstractmethod
    def has_problems_remaining(self) -> bool:
        """
        Returns:
             true iff the factory is still going to generate problems.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_problem_list(self, limit) -> [(int, NavigationProblem)]:
        """
        Yields all available problems as a list.
        Returns:
            List of (index, NavigationProblem)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def next_problem(self) -> NavigationProblem:
        """
        Yield the next problem to be used by the Gym environment.
        Returns:
            Next problem as a Problem object
        """
        raise NotImplementedError
