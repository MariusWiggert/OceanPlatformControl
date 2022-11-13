import logging
import os
from typing import Dict, List, Optional
import time

from ocean_navigation_simulator.environment.PlatformState import (
    SpatialPoint,
)

import xarray as xr
import numpy as np
import casadi as ca
from ocean_navigation_simulator.utils import units


class BathymetrySource:
    def __init__(
        self,
        casadi_cache_dict: Dict,
        source_dict: Dict,
        use_geographic_coordinate_system: Optional[bool] = True,
    ):
        """Initialize the source objects from the respective settings dicts

        Args:
            casadi_cache_dict (Dict): containing the cache settings to use in the sources for caching of 3D data
                          e.g. {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*5} for 5 days
            source_dict (Dict): _description_
            use_geographic_coordinate_system (Optional[bool], optional): _description_. Defaults to True.
        """
        # TODO: do we need time_around_x_t?
        self.logger = logging.getLogger("areana.bathymetry_source")
        self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
        self.DataArray = None
        self.source_dict = source_dict

        # Step 1: create source
        start = time.time()
        self.source_dict["casadi_cache_settings"] = casadi_cache_dict
        self.source_dict["use_geographic_coordinate_system"] = use_geographic_coordinate_system
        self.source_dict = self.instantiate_source_from_dict(source_dict)
        # Write functions
        # self.set_casadi_function()
        # self.grid_dict = self.get_grid_dict_from_xr(self.DataArray)
        self.logger.info(f"BathymetrySource: Create source({time.time() - start:.1f}s)")

    # Realizing that Datasource is highly time dependent and doesn't make any sense for static data :/

    def instantiate_source_from_dict(self, source_dict: dict):
        if source_dict["source"] == "gebco":
            self.DataArray = self.get_bathymetry_from_file()
        else:
            raise NotImplementedError(
                f"Selected source {source_dict['source']} in the BathymetrySource dict is not implemented."
            )

    def get_bathymetry_from_file(self) -> xr:
        DataArray = xr.open_dataset(self.source_dict["source_settings"]["filepath"])
        # DataArray = DataArray.rename({"latitude": "lat", "longitude": "lon"})
        return DataArray

    # TODO whole function
    # def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
    #     """DataSource specific function to initialize the casadi functions needed.
    #     # Note: the input to the casadi function needs to be an array of the form np.array([posix time, lat, lon])
    #     Args:
    #       grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
    #       array:    xarray object containing the sub-setted data for the next cached round
    #     """

    #     self.u_curr_func = ca.interpolant(
    #         "u_curr", "linear", grid, array["water_u"].values.ravel(order="F")
    #     )
    #     self.v_curr_func = ca.interpolant(
    #         "v_curr", "linear", grid, array["water_v"].values.ravel(order="F")
    #     )

    def get_data_at_point(self, spatial_point: SpatialPoint) -> float:
        # TODO seems like get_ground_truth (ocean_field.get_ground_truth), (u, v)
        pass

    def get_area(
        self,
        x_interval: List[float],
        y_interval: List[float],
        spatial_resolution: Optional[float] = None,
    ) -> xr:
        """A function to receive the ground_truth for a specific area
        Args:
            x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
            y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
                spatial_resolution: spatial resolution in the same units as x and y interval
            Returns:
            data_array                    in xarray format that contains the grid and the value
        """
        pass

    def map_analytical_function_over_area(self, grids_dict: dict) -> np.array:
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
        Args:
          grids_dict: containing ranges and grids of x, y, t dimension
        Returns:
          data_tuple     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        pass

    def depth(self):
        # Create like solar_irradiance_analytical in SolarIrradianceSource.py
        pass

    def plot_data_over_area(self):
        pass

    # TODO: probably we could do this with geopy for better accuracy
    def is_on_land(self, point: SpatialPoint):
        """Helper function to check if a SpatialPoint is on the land indicated in the
            nc files as NaN (only approximate land boundaries).
            Accuracy is limited by the resolution of self.grid_dict.
        Args:
            point:    SpatialPoint object where to check if it is on land
        Returns:
            bool:     True if on land and false otherwise
        """
        if not self.grid_dict["x_grid"].min() < point.lon.deg < self.grid_dict["x_grid"].max():
            raise ValueError(
                f'Point {point} is not inside x_dict {self.grid_dict["x_grid"][[0,-1]]}'
            )
        if not self.grid_dict["y_grid"].min() < point.lat.deg < self.grid_dict["y_grid"].max():
            raise ValueError(
                f'Point {point} is not inside y_grid {self.grid_dict["y_grid"][[0,-1]]}'
            )

        x_idx = (np.abs(self.grid_dict["x_grid"] - point.lon.deg)).argmin()
        y_idx = (np.abs(self.grid_dict["y_grid"] - point.lat.deg)).argmin()
        return self.grid_dict["spatial_land_mask"][y_idx, x_idx]

    # TODO: probably we could do this with geopy for better accuracy
    def distance_to_land(self, point: SpatialPoint) -> units.Distance:
        """
            Helper function to get the distance of a SpatialPoint to land.
            Accuracy is limited by the resolution of self.grid_dict.
        Args:
            point:    SpatialPoint object where to calculate distance to land
        Returns:
            units.Distance:     Distance to closest land
        """
        if not self.grid_dict["x_grid"].min() < point.lon.deg < self.grid_dict["x_grid"].max():
            raise ValueError(
                f'Point {point} is not inside x_dict {self.grid_dict["x_grid"][[0,-1]]}'
            )
        if not self.grid_dict["y_grid"].min() < point.lat.deg < self.grid_dict["y_grid"].max():
            raise ValueError(
                f'Point {point} is not inside y_grid {self.grid_dict["y_grid"][[0,-1]]}'
            )

        lon1, lat1 = np.meshgrid(
            point.lon.deg * np.ones_like(self.grid_dict["x_grid"]),
            point.lat.deg * np.ones_like(self.grid_dict["y_grid"]),
        )
        lon2, lat2 = np.meshgrid(self.grid_dict["x_grid"], self.grid_dict["y_grid"])

        distances = np.vectorize(units.haversine_rad_from_deg)(lon1, lat1, lon2, lat2)
        land_distances = np.where(self.grid_dict["spatial_land_mask"], distances, np.inf)

        return units.Distance(rad=land_distances.min())

    def __del__(self):
        """Helper function to delete the existing casadi functions."""
        # TODO: is this the casadi function?
        # print('__del__ called in OceanCurrentSource')
        del self.DataArray
        pass
