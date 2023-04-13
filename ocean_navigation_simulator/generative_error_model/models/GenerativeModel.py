import datetime
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import xarray as xr


class GenerativeModel(ABC):
    """
    Abstract class which describes the functionality of what a generative
    model needs to have.
    """

    @abstractmethod
    def reset(self, rng: Optional[np.random.default_rng]) -> None:
        """Initializes the model with a new random number generator."""

    @abstractmethod
    def get_noise(
        self,
        lon_locs: List[float],
        lat_locs: List[float],
        t_interval: List[datetime.datetime],
        t_origin: datetime.datetime,
    ) -> xr.Dataset:
        """Returns the noise field given a specific rng."""
