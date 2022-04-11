import abc
import numpy as np
from ocean_navigation_simulator.env.controllers import simulator_data


class Controller(abc.ABC):
    """
    Interface for controllers.
    """

    @abc.abstractmethod
    def get_action(self, observation: simulator_data.SimulatorObservation) -> np.ndarray:
        """
        Given an observation, outputs the controller's next action
        Args:
          observation: observed state from simulator or other source (i.e. observer, other controller)
        Returns:
          Controller's next action as a numpy array.
        """

    # TODO: other methods needed?
