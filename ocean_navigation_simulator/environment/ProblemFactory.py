import abc

from ocean_navigation_simulator.environment.Problem import Problem

"""
Problem Factories generate Problems. They can either be completely random and infinite
or they use a pre-generated finite list. For reproducibility a seed can be fed at generation.
"""

class ProblemFactory(abc.ABC):
    """
    Interface for a problem generator.
    """
    def __init__(self, seed: int = None):
        """
        Set the seed for an RNG in the ProblemFactory.
        """
        self.seed = seed

    @abc.abstractmethod
    def next_problem(self) -> Problem:
        """
        Yield the next problem to be used by the Gym environment.
        Returns:
            Next problem as a Problem object
        """


    @abc.abstractmethod
    def has_problems_remaining(self) -> bool:
        """
        Returns:
             true iff the factory is still going to generate problems.
        """
