from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

from ocean_navigation_simulator.env.PlatformState import SpatioTemporalPoint
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector


class OceanCurrentModel(ABC):
    def __init__(self):
        self.measurement_locations, self.errors = None, None
        self.reset()

    # @abstractmethod
    # def get_mean_predictions(self, x_interval: List[float], y_interval: List[float],
    #                          t_interval: List[Union[datetime.datetime, int]]) -> xarray:
    #     pass
    #
    # @abstractmethod
    # def get_std_predictions(self, x_interval: List[float], y_interval: List[float],
    #                         t_interval: List[Union[datetime.datetime, int]]) -> xarray:
    #     pass

    def get_prediction(self, lon: float, lat: float, time: datetime.datetime) \
            -> Union[OceanCurrentVector, Tuple[OceanCurrentVector, OceanCurrentVector]]:
        res = self.get_predictions(np.array([lon, lat, time]))
        return (res[0][0], res[1][0]) if (res is tuple) is not None else res[0]

    @abstractmethod
    def get_predictions(self, locations: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        pass

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        self.measurement_locations = list()
        self.errors = list()

    def observe(self, measurement: SpatioTemporalPoint, error: OceanCurrentVector) -> None:
        self.measurement_locations.append(np.array(measurement))
        self.errors.append(np.array(error))
