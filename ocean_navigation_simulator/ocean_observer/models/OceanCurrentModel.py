from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

from ocean_navigation_simulator.environment.PlatformState import SpatioTemporalPoint
from ocean_navigation_simulator.environment.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector


class OceanCurrentModel(ABC):
    """Abstract class that describe what the OceanCurrentModel used by the observer to predict current should be able to
    do """

    def __init__(self):
        self.measurement_locations, self.measured_current_errors = None, None
        self.reset()

    def get_prediction(self, lon: float, lat: float, time: datetime.datetime) \
            -> Union[OceanCurrentVector, Tuple[OceanCurrentVector, OceanCurrentVector]]:
        """Compute the predictions for a given point

        Args:
            lon: [in degree] longitude coordinate of the point
            lat: [in degree] latitude coordinate of the point
            time: time of the point

        Returns:
            The prediction of the model for the given point (including the std if provided)
        """
        res = self.get_predictions(np.array([[lon, lat, time]]))
        return (res[0][0], res[1][0]) if (res is tuple) is not None else res[0]

    @abstractmethod
    def get_predictions(self, locations: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """ Compute the predictions for the oceancurrents locations and the std dev of these points if supported by the
        kind of models

        Args:
            locations: (N,3) ndarray that contains all the points we want to predict. Each point should be described by:
                       (lon in degree, lat in degree, time in datetime.datetime format)
        Returns:
            the predictions (N,3) and the std of these predictions (N,3) if supported
        """
        pass

    @abstractmethod
    def fit(self) -> None:
        """Fitting the model using the observations recorded
        """
        pass

    def reset(self) -> None:
        """Reset by deleting the observations by default
        """
        self.measurement_locations = list()
        self.measured_current_errors = list()

    def observe(self, measurement_location: SpatioTemporalPoint, measured_current_error: OceanCurrentVector) -> None:
        """Add an observation at the position measurement that will be used when fitting the model.
        
        Args:
            measurement_location: Position of the observation
            measured_current_error: difference between the forecast and the ground truth currents at that position
        """
        self.measurement_locations.append(np.array(measurement_location))
        self.measured_current_errors.append(np.array(measured_current_error))
