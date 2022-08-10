from abc import ABC, abstractmethod
from ocean_navigation_simulator.utils import units

import datetime


class GenerativeModel(ABC):
    """
    Abstract class which describes the functionality of what a generative
    model needs to have.
    """
    
    def __init__(self):
        self.model = model


    @abstractmethod
    def reset(self, rng) -> None:
        """Initializes the model with a new random number generator."""


    def get_noise_field(self, x: units.Distance, y: units.Distance,
        elapsed_time: datetime.timedelta) -> None:
        """Returns the noise field given a specific rng."""