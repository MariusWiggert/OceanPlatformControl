from __future__ import annotations

import datetime
from typing import List, Optional, Union

import xarray

from ocean_navigation_simulator.environment.Arena import ArenaObservation

# TODO: change to use loggers


class NoObserver:
    """Class that represent the observer. It will receive observations and is then responsible to return predictions for
    the given areas.
    """

    def __init__(self):
        """Create the observer object
        Args: None
        """
        self.forecast_data_source = None

    def get_data_over_area(
        self,
        x_interval: List[float],
        y_interval: List[float],
        t_interval: List[Union[datetime.datetime, int]],
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
        throw_exceptions: Optional[bool] = True,
    ) -> xarray:
        """Computes the xarray dataset that contains the prediction errors, the new forecasts (water_u & water_v), the
        old forecasts (renamed: initial_forecast_u & initial_forecast_v) and also std_error_u & std_error_v for the u
        and v directions respectively if the OceanCurrentModel output these values
        Args:
            x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
            y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
            t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime
            spatial_resolution: spatial resolution in the same units as x and y interval
            temporal_resolution: temporal resolution in seconds
        Returns:
            the computed xarray
        """
        forecasts = self.forecast_data_source.get_data_over_area(
            x_interval, y_interval, t_interval, spatial_resolution, temporal_resolution,
            throw_exceptions=throw_exceptions,
        )

        return forecasts

    def get_data_at_point(self, lon: float, lat: float, time: datetime.datetime):
        return self.forecast_data_source.get_data_at_point(lon, lat, time)

    def fit(self) -> None:
        """Fit the inner prediction model using the observations recorded by the observer"""
        pass

    def reset(self) -> None:
        """Reset the observer and its prediction model"""
        pass

    def observe(self, arena_observation: ArenaObservation) -> None:
        """Add the error at the position of the arena_observation and also dave the forecast if it has not been done
        already
        Args:
            arena_observation: Observation where we compute the error and add to the observations
        """
        if self.forecast_data_source is None:
            self.forecast_data_source = arena_observation.forecast_data_source

    # Forwarding functions as it replaces the forecast_data_source
    def check_for_most_recent_fmrc_dataframe(self, time: datetime.datetime) -> int:
        """Helper function to check update the self.OceanCurrent if a new forecast is available at
        the specified input time.
        Args:
          time: datetime object
        """
        return self.forecast_data_source.check_for_most_recent_fmrc_dataframe(time=time)
