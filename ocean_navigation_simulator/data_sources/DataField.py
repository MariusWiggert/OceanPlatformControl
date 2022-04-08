import abc
import datetime
from typing import List, NamedTuple, Sequence, Callable, Optional
import numpy as np

# import gin # We don't use gin because it doesn't work well with the C3 Data Types. Hence, we use settings_dicts.
import xarray as xr


class DataField(abc.ABC):
    """Abstract class for lookups in an DataField.
  Both point-based lookup (for simulation) and spatio-temporal interval lookup (for planning)
  of both ground_truth and forecasted Data (e.g. Ocean currents, solar radiation, seaweed growth)
  """

    @abc.abstractmethod
    def get_forecast(self, point: List[float], time: datetime.datetime):
        """Returns forecast at a point in the field.
    Args:
      point: Point in the respective used coordinate system (lat, lon for geospherical or unitless for examples)
      time: absolute datetime object
    Returns:
      A Field Data for the position in the DataField (Vector or other).
    """

    @abc.abstractmethod
    def get_forecast_area(self, x_interval: List[float], y_interval: List[float], t_interval: List[datetime.datetime],
                          spatial_resolution: float, temporal_resolution) -> xr:
        """A function to receive the forecast for a specific area over a time interval.
    Args:
      x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
      y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
      t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime
      spatial_resolution: spatial resolution in the same units as x and y interval
      temporal_resolution: temporal resolution TODO: how which units?
    Returns:
      data_array                    in xarray format that contains the grid and the values
    """

    @abc.abstractmethod
    def get_ground_truth(self, point: List[float], time: datetime.datetime):
        """Returns true data at a point in the field.
    Args:
      point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
      time: absolute datetime object
    Returns:
      A Field Data for the position in the DataField (Vector or other).
    """

    @abc.abstractmethod
    def get_ground_truth_area(self, x_interval: List[float], y_interval: List[float],
                              t_interval: List[datetime.datetime],
                              spatial_resolution: float, temporal_resolution) -> xr:
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
