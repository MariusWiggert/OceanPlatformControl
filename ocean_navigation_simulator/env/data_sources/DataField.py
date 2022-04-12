import abc
import datetime
from typing import List, NamedTuple, Sequence, Callable, Optional
import numpy as np
import warnings
import ocean_navigation_simulator.env.utils.units as units

# import gin # We don't use gin because it doesn't work well with the C3 Data Types. Hence, we use settings_dicts.
import xarray as xr

# TODO: need a marker to output lat lon or x and y (for analytical currents)
# => also need to be able to switch that in the documents...


class DataField(abc.ABC):
    """Abstract class for lookups in an DataField.
  Both point-based lookup (for simulation) and spatio-temporal interval lookup (for planning)
  of both ground_truth and forecasted Data (e.g. Ocean currents, solar radiation, seaweed growth)
  """
    def __init__(self, hindcast_source_dict: dict, forecast_source_dict: Optional[dict] = None):
        """Initialize the source objects from the respective settings dicts.
        Args:
          forecast_source_dict and hindcast_source_dict
           Both are dicts with four keys:
             'field' the kind of field the should be created e.g. OceanCurrent or SolarIrradiance
             'subset_time_buffer_in_s' specifying the buffer applied to the time-interval when sub-setting an area
             'casadi_cache_settings': e.g. {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*12} for caching of 3D data
             'source' see the class respectively for the available options
             'source_settings' dict that contains the specific settings required for the selected 'source'. See classes.
        """
        # Step 1: instantiate OceanCurrentSources from their respective dicts
        self.hindcast_data_source = self.instantiate_source_from_dict(hindcast_source_dict)
        if forecast_source_dict is None:
            print("Forecast is the same as Hindcast for {}.".format(hindcast_source_dict['field']))
            self.forecast_data_source = self.hindcast_data_source

    def get_forecast(self, point: List[float], time: datetime.datetime):
        """Returns forecast at a point in the field.
        Args:
          point: Point in the respective used coordinate system (lat, lon for geospherical or unitless for examples)
          time: absolute datetime object
        Returns:
          A Field Data for the position in the DataField (Vector or other).
        """
        return self.forecast_data_source.get_data_at_point(point, time)

    def get_forecast_area(self, x_interval: List[float], y_interval: List[float], t_interval: List[datetime.datetime],
                          spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None) -> xr:
        """A function to receive the forecast for a specific area over a time interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime
          spatial_resolution: spatial resolution in the same units as x and y interval
          temporal_resolution: temporal resolution in seconds
        Returns:
          data_array                    in xarray format that contains the grid and the values
        """
        return self.forecast_data_source.get_data_over_area(x_interval, y_interval, t_interval,
                                                            spatial_resolution, temporal_resolution)

    def get_ground_truth(self, point: List[float], time: datetime.datetime):
        """Returns true data at a point in the field.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          A Field Data for the position in the DataField (Vector or other).
        """
        return self.hindcast_data_source.get_data_at_point(point, time)

    def get_ground_truth_area(self, x_interval: List[float], y_interval: List[float],
                              t_interval: List[datetime.datetime],
                              spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None) -> xr:
        """A function to receive the ground_truth for a specific area over a time interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime
          spatial_resolution: spatial resolution in the same units as x and y interval
          temporal_resolution: temporal resolution in seconds
        Returns:
          data_array                    in xarray format that contains the grid and the values
        """
        return self.hindcast_data_source.get_data_over_area(x_interval, y_interval, t_interval,
                                                            spatial_resolution, temporal_resolution)

    @staticmethod
    @abc.abstractmethod
    def instantiate_source_from_dict(source_dict: dict):
        """Function to instantiate the source objects from a field."""
        raise NotImplementedError

