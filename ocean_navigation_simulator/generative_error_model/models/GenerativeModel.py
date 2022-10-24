from abc import ABC, abstractmethod
from typing import Optional, List, Union
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
                  lon_locs: List[float],
                  lat_locs: List[float],
                  t_locs: List[datetime.datetime]) -> xr.Dataset:
        """Returns the noise field given a specific rng."""
