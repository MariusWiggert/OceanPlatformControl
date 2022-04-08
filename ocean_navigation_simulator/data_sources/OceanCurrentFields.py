import abc
import datetime
from typing import List, NamedTuple, Sequence, Optional
import ocean_navigation_simulator.utils.units as units
from ocean_navigation_simulator.data_sources.DataField import DataField
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.data_sources.OceanCurrentVector import OceanCurrentVector
import jax
from jax import numpy as jnp
import numpy as np
import xarray as xr
from geopy.point import Point as GeoPoint


class OceanCurrentField(DataField):
    """Class holding two data sources, the forecast and hindcast current sources
  """

    def __init__(self, forecast_data_source: OceanCurrentSource, hindcast_data_source: OceanCurrentSource):
        # save them as class variables
        self.forecast_data_source = forecast_data_source
        self.hindcast_data_source = hindcast_data_source

    def get_forecast(self, point: List[float], time: datetime.datetime) -> OceanCurrentVector:
        """Returns forecast at a point in the field.
        Args:
          point: Point in the respective used coordinate system (lat, lon for geospherical or unitless for examples)
          time: absolute datetime object
        Returns:
          A Field Data for the position in the DataField (Vector or other).
        """
        return self.forecast_data_source.get_currents_at_point(point, time)

    def get_forecast_area(self, x_interval: List[float], y_interval: List[float], t_interval: List[datetime.datetime],
                          spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None) -> xr:
        """A function to receive the forecast for a specific area over a time interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime
          spatial_resolution: spatial resolution in the same units as x and y interval
          temporal_resolution: optional temporal resolution in seconds TODO: how which units?
        Returns:
          data_array                    in xarray format that contains the grid and the values
        """
        # Step 1: get the raw sub-setted xarray
        current_array = self.forecast_data_source.get_currents_over_area(x_interval, y_interval, t_interval)

        # Step 2: perform interpolation to a specific resolution if requested
        if spatial_resolution is not None or temporal_resolution is not None:
            current_array = self.interpolate_in_space_and_time(current_array, spatial_resolution, temporal_resolution)

        return current_array

    def get_ground_truth(self, point: List[float], time: datetime.datetime) -> OceanCurrentVector:
        """Returns true data at a point in the field.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          A Field Data for the position in the DataField (Vector or other).
        """

    def get_ground_truth_area(self, x_interval: List[float], y_interval: List[float],
                              t_interval: List[datetime.datetime],
                              spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None) -> xr:
        """A function to receive the ground_truth for a specific area over a time interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime
          spatial_resolution: spatial resolution in the same units as x and y interval
          temporal_resolution: temporal resolution TODO: how which units?
        Returns:
          data_array                    in xarray format that contains the grid and the values
        """

        # Step 1: get the raw sub-setted xarray
        current_array = self.hindcast_data_source.get_currents_over_area(x_interval, y_interval, t_interval)

        # Step 2: perform interpolation to a specific resolution if requested
        if spatial_resolution is not None or temporal_resolution is not None:
            current_array = self.interpolate_in_space_and_time(current_array, spatial_resolution, temporal_resolution)

        return current_array




