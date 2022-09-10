import abc

from ocean_navigation_simulator.environment.Problem import Problem


class ProblemFactory(abc.ABC):
    """
    Interface for a problem generator.
    """

    @abc.abstractmethod
    def next_problem(self) -> Problem:
        """
        Yield the next problem to be used by the Gym environment.
        Returns:
            Next problem as a Problem object
        """

    # @abc.abstractmethod
    # def seed(self, seed: int = None):
    #     """
    #     Set the seed for an RNG in the ProblemFactory.
    #     """

    @abc.abstractmethod
    def has_problems_remaining(self) -> bool:
        """
        Returns:
             true iff the factory is still going to generate problems.
        """
