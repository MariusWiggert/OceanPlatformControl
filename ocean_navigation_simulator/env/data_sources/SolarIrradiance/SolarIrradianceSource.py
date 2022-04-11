import abc
import datetime
from typing import List, NamedTuple, Sequence, AnyStr, Optional, Tuple
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
from ocean_navigation_simulator.env.data_sources.DataSources import DataSource, AnalyticalSource, XarraySource
from ocean_navigation_simulator.env.data_sources.SolarIrradiance.solar_rad import solar_rad
import datetime
import matplotlib.pyplot as plt
from typing import List, NamedTuple, Sequence, AnyStr, Optional, Tuple
import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from functools import partial
import xarray as xr
import jax
from ocean_navigation_simulator import utils
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.data_sources.DataSources import DataSource, AnalyticalSource
import abc


class SolarIrradianceSource(DataSource):
    """Base class for the Solar Irradiance data sources."""

    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """

        self.solar_rad_casadi = ca.interpolant('irradiance', 'linear', grid, array['solar_irradiance'].values.ravel(order='F'))


class AnalyticalSolarIrradiance(AnalyticalSource, SolarIrradianceSource):
    """Data Source Object that accesses and manages one or many HYCOM files as source."""
    def __init__(self, source_config_dict):
        super().__init__(source_config_dict)
        self.solar_rad_casadi = None

    def solar_irradiance_analytical(self, point: List[float], time: float):
        """To be implemented in the child class. Note only for 2D currently."""
        return solar_rad(time, point[1], point[0])

    def create_xarray(self, grid_dict: dict, solar_irradiance: jnp.array) -> xr:
        """Function to create an xarray from the data tuple and grid dict
            Args:
              data_tuple: tuple containing (data_u, data_v)
              grid_dict: containing ranges and grids of x, y, t dimension
            Returns:
              xr     an xarray containing both the grid and data
            """
        # make a xarray object out of it
        return xr.Dataset(
            dict(solar_irradiance=(["time", "lat", "lon"], solar_irradiance)),
            coords=dict(lon=grid_dict['x_grid'], lat=grid_dict['y_grid'],
                        time=np.round(np.array(grid_dict['t_grid']) * 1000, 0).astype('datetime64[ms]')))

    # TODO: check out how I can feed vectorized into the solar_rad function (see Nisha's code) or make solar_rad jax.numpy ready
    def map_analytical_function_over_area(self, states, grid_dict):
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
            Args:
              states: jax.numpy array containing the desired spatial grid as states
              grid_dict: containing ranges and grids of x, y, t dimension
            Returns:
              data_tuple     containing the data in tuple format as numpy array (not yet in xarray form)
            """
        # Step 2: Map analytical function over the area
        solar_irradiance = lambda t: hj.utils.multivmap(partial(self.solar_irradiance_analytical, time=t), np.arange(2))(
            jnp.transpose(states, axes=[1, 0, 2]))
        # execute over temporal_vector
        solar_irradiance_data = jax.vmap(solar_irradiance)(grid_dict['t_grid'])
        return solar_irradiance_data

    def get_data_at_point(self, point: List[float], time: datetime) -> float:
        """Function to get the data at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          float of the solar irradiance in W/m^2
          """
        return self.solar_irradiance_analytical(point, time.timestamp())







