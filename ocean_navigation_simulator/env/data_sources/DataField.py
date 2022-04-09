import abc
import datetime
from typing import List, NamedTuple, Sequence, Callable, Optional
import numpy as np
import warnings
import ocean_navigation_simulator.env.utils.units as units

# import gin # We don't use gin because it doesn't work well with the C3 Data Types. Hence, we use settings_dicts.
import xarray as xr


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
             'source' in {opendap, hindcast_files, forecast_files}
             'subset_time_buffer_in_s' specifying the buffer applied to the time-interval when sub-setting an area
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
        return self.forecast_data_source.get_currents_at_point(point, time)

    def get_forecast_area(self, x_interval: List[float], y_interval: List[float], t_interval: List[datetime.datetime],
                          spatial_resolution: float, temporal_resolution) -> xr:
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
        return self.forecast_data_source.get_currents_over_area(x_interval, y_interval, t_interval,
                                                                spatial_resolution, temporal_resolution)

    def get_ground_truth(self, point: List[float], time: datetime.datetime):
        """Returns true data at a point in the field.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          A Field Data for the position in the DataField (Vector or other).
        """
        return self.hindcast_data_source.get_currents_at_point(point, time)

    def get_ground_truth_area(self, x_interval: List[float], y_interval: List[float],
                              t_interval: List[datetime.datetime],
                              spatial_resolution: float, temporal_resolution) -> xr:
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
        return self.hindcast_data_source.get_currents_over_area(x_interval, y_interval, t_interval,
                                                                spatial_resolution, temporal_resolution)

    @staticmethod
    def interpolate_in_space_and_time(array: xr, spatial_resolution: Optional[float],
                                      temporal_resolution: Optional[float]) -> xr:
        """Helper function for temporal and spatial interpolation"""
        # Run temporal interpolation
        if temporal_resolution is not None:
            time_grid = np.arange(start=array['time'][0].data, stop=array['time'][-1].data,
                                  step=np.timedelta64(temporal_resolution, 's'))
            array = array.interp(time=time_grid, method='linear')

        # Run spatial interpolation
        if spatial_resolution is not None:
            lat_grid = np.arange(start=array['lat'][0].data, stop=array['lat'][-1].data,
                                 step=spatial_resolution)
            lon_grid = np.arange(start=array['lon'][0].data, stop=array['lon'][-1].data,
                                 step=spatial_resolution)
            array = array.interp(
                lon=lon_grid,
                lat=lat_grid, method='linear')

        return array

    @staticmethod
    def array_subsetting_sanity_check(array:xr, x_interval: List[float], y_interval: List[float],
                               t_interval: List[datetime.datetime]):
        """Advanced Check if admissible subset and warning of partially being out of bound in space or time."""
        # Step 1: collateral check is any dimension 0?
        if 0 in array['water_u'].data.shape:
            # check which dimension for more informative errors
            if array['water_u'].data.shape[0] == 0:
                raise ValueError("None of the requested t_interval is in the file.")
            else:
                raise ValueError("None of the requested spatial area is in the file.")
        if units.get_datetime_from_np64(array.coords['time'].data[0]) > t_interval[0]:
            raise ValueError("The starting time {} is not in the array.".format(t_interval[0]))

        # Step 2: Data partially not in the array check
        if array.coords['lat'].data[0] > y_interval[0] or array.coords['lat'].data[-1] < y_interval[1]:
            warnings.warn("Part of the y requested area is outside of file.", RuntimeWarning)
        if array.coords['lon'].data[0] > x_interval[0] or array.coords['lon'].data[-1] < x_interval[1]:
            warnings.warn("Part of the x requested area is outside of file.", RuntimeWarning)
        if units.get_datetime_from_np64(array.coords['time'].data[-1]) < t_interval[1]:
            warnings.warn("The final time is not part of the subset.".format(t_interval[1]), RuntimeWarning)

    @staticmethod
    @abc.abstractmethod
    def instantiate_source_from_dict(source_dict: dict):
        """Function to instantiate the source objects from a field."""
        raise NotImplementedError
