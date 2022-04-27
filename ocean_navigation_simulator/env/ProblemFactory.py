import abc
from ocean_navigation_simulator.env.Problem import Problem


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

    # TODO: other methods needed?
