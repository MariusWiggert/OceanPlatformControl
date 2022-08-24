from ocean_navigation_simulator.generative_error_model.Problem import Problem

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import datetime
import xarray as xr
import numpy as np


class GenerativeModel(ABC):
    """
    Abstract class which describes the functionality of what a generative
    model needs to have.
    """

    @abstractmethod
    def reset(self, rng: Optional[np.random.default_rng]) -> None:
        """Initializes the model with a new random number generator."""

    @abstractmethod
    def get_noise(self, problem: Problem) -> xr.Dataset:
        """Returns the noise field given a specific rng."""
