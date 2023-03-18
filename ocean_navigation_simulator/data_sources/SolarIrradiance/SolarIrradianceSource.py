import datetime
import logging
import time
from typing import List, Optional, Union

import casadi as ca
import numpy as np
import xarray as xr

from ocean_navigation_simulator.data_sources.DataSource import (
    AnalyticalSource,
    DataSource,
)
from ocean_navigation_simulator.data_sources.SolarIrradiance.solar_rad import (
    solar_rad,
    solar_rad_ca,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatioTemporalPoint,
)


class SolarIrradianceSource(DataSource):
    """Base class for the Solar Irradiance data sources.
    Note: the output is always in the units W/m^2
    """

    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        Note: the input to the casadi function needs to be an array of the form np.array([posix time, lat, lon])
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """
        self.solar_rad_casadi = ca.interpolant(
            "irradiance", "linear", grid, array["solar_irradiance"].values.ravel(order="F")
        )


class AnalyticalSolarIrradiance_w_caching(AnalyticalSource, SolarIrradianceSource):
    """Data Source Object that accesses and manages one or many HYCOM files as source."""

    def __init__(self, source_config_dict):
        """Dictionary with the three top level keys:
         'field' the kind of field the should be created, here SolarIrradiance
         'source' in {analytical} (currently no others implemented)
         'source_settings':{
            'x_domain': [-180, 180],        # in degree lat, lon
            'y_domain': [-90, 90],          # in degree lat, lon
            'temporal_domain': [datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
                                datetime.datetime(2023, 1, 10, 0, 0, 0, tzinfo=datetime.timezone.utc)],
            'spatial_resolution': 0.1,      # in degree lat, lon
            'temporal_resolution': 3600,    # in seconds
        }
        """
        super().__init__(source_config_dict)
        # initialize logger
        self.logger = logging.getLogger("arena.solar_field.analytical_source")
        self.solar_rad_casadi = None

    @staticmethod
    def solar_irradiance_analytical(
        lon: Union[float, np.array], lat: Union[float, np.array], posix_time: Union[float, np.array]
    ) -> Union[float, np.array]:
        """Calculating the solar Irradiance at a specific lat, lon point at posix_time.
        Note: this can be used for floats as input or np.arrays which are all the same shape.
        Args:
            lon: longitude in degree
            lat: latitude in degree
            posix_time: POSIX time
        Returns:
            solar_irradiance     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        return solar_rad(posix_time, lat, lon)

    def create_xarray(self, grid_dict: dict, solar_irradiance: np.array) -> xr:
        """Function to create an xarray from the data tuple and grid dict
        Args:
          solar_irradiance: numpy array [T, Y, X] of data
          grid_dict: containing ranges and grids of x, y, t dimension
        Returns:
          xr     an xarray containing both the grid and data
        """
        # make a xarray object out of it
        array = xr.Dataset(
            dict(solar_irradiance=(["time", "lat", "lon"], solar_irradiance)),
            coords=dict(
                lon=grid_dict["x_grid"],
                lat=grid_dict["y_grid"],
                time=np.round(np.array(grid_dict["t_grid"]) * 1000, 0).astype("datetime64[ms]"),
            ),
        )
        # add the units
        array["solar_irradiance"].attrs = {"units": "W/m^2"}
        return array

    def map_analytical_function_over_area(self, grids_dict: dict) -> np.array:
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
        Args:
          grids_dict: containing ranges and grids of x, y, t dimension
        Returns:
          data_tuple     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        # Step 1: Create the meshgrid numpy matrices for each coordinate
        LAT, TIMES, LON = np.meshgrid(
            grids_dict["y_grid"], grids_dict["t_grid"], grids_dict["x_grid"]
        )

        # Step 2: Feed the arrays into the solar radiation function and return the np.array
        return self.solar_irradiance_analytical(lon=LON, lat=LAT, posix_time=TIMES)

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> float:
        """Function to get the data at a specific point.
        Args:
          spatio_temporal_point: SpatioTemporalPoint in the respective used coordinate system geospherical or unitless
        Returns:
          float of the solar irradiance in W/m^2
        """

        return self.solar_irradiance_analytical(
            lon=spatio_temporal_point.lon.deg,
            lat=spatio_temporal_point.lat.deg,
            posix_time=spatio_temporal_point.date_time.timestamp(),
        )

    def get_data_over_area(
        self,
        x_interval: List[float],
        y_interval: List[float],
        t_interval: List[Union[datetime.datetime, float]],
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
        throw_exceptions: Optional[bool] = True,
    ) -> xr:
        """Function to get the the solar data over an x, y, and t interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime or posix.
          spatial_resolution: spatial resolution in the same units as x and y interval
          temporal_resolution: temporal resolution in seconds
        Returns:
          data_array     in xarray format that contains the grid and the values (land is NaN)
        """

        start = time.time()
        # Step 0.0: if t_interval is in datetime convert to POSIX
        if isinstance(t_interval[0], datetime.datetime):
            t_interval_posix = [time.timestamp() for time in t_interval]
        else:
            t_interval_posix = t_interval
            t_interval = [
                datetime.datetime.fromtimestamp(posix, tz=datetime.timezone.utc)
                for posix in t_interval_posix
            ]

        # Get the coordinate vectors to calculate the analytical function over
        grids_dict = self.get_grid_dict(
            x_interval,
            y_interval,
            t_interval_posix,
            spatial_resolution=spatial_resolution,
            temporal_resolution=temporal_resolution,
        )

        data_tuple = self.map_analytical_function_over_area(grids_dict)

        # make an xarray object out of it
        subset = self.create_xarray(grids_dict, data_tuple)

        # Step 3: Do a sanity check for the sub-setting before it's used outside and leads to errors
        self.array_subsetting_sanity_check(subset, x_interval, y_interval, t_interval, self.logger)

        self.logger.info(
            f"SolarIrradianceSource: get Data over Area finished ({time.time() - start:.1f}s)"
        )

        return subset


class AnalyticalSolarIrradiance(AnalyticalSolarIrradiance_w_caching):
    """Data Source Object that accesses and manages one or many HYCOM files as source.
    It does not use caching and directly uses the analytical casadi function.
    """

    def __init__(self, source_config_dict):
        """Dictionary with the three top level keys:
         'field' the kind of field the should be created, here SolarIrradiance
         'source' in {analytical} (currently no others implemented)
         'source_settings':{
            'x_domain': [-180, 180],        # in degree lat, lon
            'y_domain': [-90, 90],          # in degree lat, lon
            'temporal_domain': [datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
                                datetime.datetime(2023, 1, 10, 0, 0, 0, tzinfo=datetime.timezone.utc)],
            'spatial_resolution': 0.1,      # in degree lat, lon
            'temporal_resolution': 3600,    # in seconds
        }
        """
        super().__init__(source_config_dict)
        # initialize logger
        self.logger = logging.getLogger("arena.solar_field.analytical_source")
        self.solar_rad_casadi = solar_rad_ca
        # set the self.casadi_grid_dict to full domain
        self.casadi_grid_dict = self.grid_dict

    def update_casadi_dynamics(self, state: PlatformState):
        """Passing the function because nothing needs to be updated."""
        pass


class FixedYRangeSolar(AnalyticalSolarIrradiance):
    """Simple Solar field with 0 irradiance everywhere except at in a specific y range."""

    def __init__(self, source_config_dict):
        super().__init__(source_config_dict)
        # initialize logger
        self.logger = logging.getLogger("arena.solar_field.analytical_source")

        self.y_range_solar = source_config_dict["source_settings"]["y_range_solar"]
        self.irradiance = source_config_dict["source_settings"]["irradiance"]

    def solar_irradiance_analytical(
        self,
        lon: Union[float, np.array],
        lat: Union[float, np.array],
        posix_time: Union[float, np.array],
    ) -> Union[float, np.array]:
        """Calculating the solar Irradiance at a specific lat, lon point at posix_time.
        Note: this can be used for floats as input or np.arrays which are all the same shape.
        Args:
            lon: longitude in degree
            lat: latitude in degree
            posix_time: POSIX time
        Returns:
            solar_irradiance     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        irradiance_low = np.where(lat <= self.y_range_solar[1], self.irradiance, 0.0)
        irradiance_out = np.where(self.y_range_solar[0] <= lat, irradiance_low, 0.0)
        return irradiance_out
