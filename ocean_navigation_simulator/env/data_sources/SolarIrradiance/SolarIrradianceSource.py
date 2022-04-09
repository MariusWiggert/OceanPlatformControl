import abc
import datetime
from typing import List, NamedTuple, Sequence, AnyStr, Optional
from ocean_navigation_simulator.env.utils.units import get_posix_time_from_np64, get_datetime_from_np64
from ocean_navigation_simulator.env.data_sources.DataField import DataField
import casadi as ca
import jax
from jax import numpy as jnp
import warnings
import numpy as np
import xarray as xr
import dask.array.core
import os
import ocean_navigation_simulator.utils as utils
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
from geopy.point import Point as GeoPoint
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.data_sources.DataSources import DataSource
from ocean_navigation_simulator.env.data_sources.SolarIrradiance.solar_rad import solar_rad


class SolarIrradianceSource(DataSource):
    """Base class for the Solar Irradiance data sources."""

    def __init__(self, source_config_dict: dict):
        """Function to get the OceanCurrentVector at a specific point.
        Args:
          source_config_dict: TODO: detail what needs to be specified here
          """
        super().__init__(source_config_dict)
        # Casadi functions are created and maintained here but used in the platform object
        self.solar_rad_casadi = None

    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """

        self.solar_rad_casadi = ca.interpolant('irradiance', 'linear', grid, array['irradiance'].values.ravel(order='F'))

    def get_data_over_area(self, x_interval: List[float], y_interval: List[float],
                           t_interval: List[datetime.datetime],
                           spatial_resolution: Optional[float] = None,
                           temporal_resolution: Optional[float] = None) -> xr:
        """Function to get the the raw current data over an x, y, and t interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime
          spatial_resolution: spatial resolution in the same units as x and y interval
          temporal_resolution: temporal resolution in seconds
        Returns:
          data_array     in xarray format that contains the grid and the values (land is NaN)
        """
        # Step 1: Subset and interpolate the xarray accordingly in the DataSource Class
        subset = super().get_data_over_area(x_interval, y_interval, t_interval, spatial_resolution, temporal_resolution)

        return subset


class AnalyticalSolarIrradiance(SolarIrradianceSourceXarray):
    """Data Source Object that accesses and manages one or many HYCOM files as source."""
    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """

        self.solar_rad_casadi = solar_rad

