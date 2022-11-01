from ocean_navigation_simulator.generative_error_model.models.Problem import Problem

from abc import ABC, abstractmethod
from typing import Optional, List
import xarray as xr
import numpy as np
import datetime


class GenerativeModel(ABC):
    """
    Abstract class which describes the functionality of what a generative
    model needs to have.
    """

    @abstractmethod
    def reset(self, rng: Optional[np.random.default_rng]) -> None:
        """Initializes the model with a new random number generator."""

    @abstractmethod
    def get_noise(self,
                  x_interval: List[float],
                  y_interval: List[float],
                  t_interval: List[datetime.datetime],
                  spatial_resolution: Optional[float],
                  temporal_resolution: Optional[float]) -> xr.Dataset:
        """Returns the noise field given a specific rng."""
