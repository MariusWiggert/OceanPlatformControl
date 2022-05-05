import abc
import datetime
from typing import List, NamedTuple, Sequence, Callable, Optional, Dict
from ocean_navigation_simulator.env.PlatformState import SpatioTemporalPoint
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
    def __init__(self, sim_cache_dict: Dict, hindcast_source_dict: Dict, forecast_source_dict: Optional[Dict] = None,
                 use_geographic_coordinate_system: Optional[bool] = True):
        """Initialize the source objects from the respective settings dicts.
        Args:
          sim_cache_dict: containing the cache settings to use in the sources for caching of 3D data
                          e.g. {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*5} for 5 days

          forecast_source_dict and hindcast_source_dict
           Both are dicts with four keys:
             'field' the kind of field the should be created e.g. OceanCurrent or SolarIrradiance
             'source' in {opendap, hindcast_files, forecast_files}
             'subset_time_buffer_in_s' specifying the buffer applied to the time-interval when sub-setting an area
             'source_settings' dict that contains the specific settings required for the selected 'source'. See classes.
        """
        # Step 1: instantiate OceanCurrentSources from their respective dicts
        hindcast_source_dict['casadi_cache_settings'] = sim_cache_dict
        hindcast_source_dict['use_geographic_coordinate_system'] = use_geographic_coordinate_system
        self.hindcast_data_source = self.instantiate_source_from_dict(hindcast_source_dict)
        if forecast_source_dict is None:
            #print("Forecast is the same as Hindcast for {}.".format(hindcast_source_dict['field']))
            self.forecast_data_source = self.hindcast_data_source
        else:
            forecast_source_dict['casadi_cache_settings'] = sim_cache_dict
            forecast_source_dict['use_geographic_coordinate_system'] = use_geographic_coordinate_system
            self.forecast_data_source = self.instantiate_source_from_dict(forecast_source_dict)

    def get_forecast(self, spatio_temporal_point: SpatioTemporalPoint):
        """Returns forecast at a point in the field.
        Args:
          spatio_temporal_point: SpatioTemporalPoint in the respective used coordinate system geospherical or unitless
        Returns:
          A Field Data for the position in the DataField (Vector or Data Object).
        """
        return self.forecast_data_source.get_data_at_point(spatio_temporal_point)

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
                                                            spatial_resolution=spatial_resolution,
                                                            temporal_resolution=temporal_resolution)

    def get_ground_truth(self, spatio_temporal_point: SpatioTemporalPoint):
        """Returns true data at a point in the field.
        Args:
          spatio_temporal_point: SpatioTemporalPoint in the respective used coordinate system geospherical or unitless
        Returns:
          A Field Data for the position in the DataField (Vector or Data Object).
        """
        return self.hindcast_data_source.get_data_at_point(spatio_temporal_point)

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

    def plot_forecast_at_time_over_area(self, time: datetime.datetime,
                                        x_interval: List[float], y_interval: List[float]):
        self.forecast_data_source.plot_data_at_time_over_area(time=time, x_interval=x_interval, y_interval=y_interval)

    def plot_true_at_time_over_area(self, time: datetime.datetime,
                                        x_interval: List[float], y_interval: List[float]):
        self.hindcast_data_source.plot_data_at_time_over_area(time=time, x_interval=x_interval, y_interval=y_interval)



