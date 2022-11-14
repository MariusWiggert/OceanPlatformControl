import logging
import os
import time
from typing import AnyStr, Dict, List, Optional

import casadi as ca
import numpy as np
import xarray as xr

from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
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
        self.DataArray = None  # Xarray containing raw data
        self.elevation_func = None  # Casadi function
        self.grid_dict, self.casadi_grid_dict = [None] * 2
        self.source_dict = source_dict

        # Step 1: create source
        start = time.time()
        self.source_dict["casadi_cache_settings"] = casadi_cache_dict
        self.source_dict["use_geographic_coordinate_system"] = use_geographic_coordinate_system
        self.instantiate_source_from_dict(source_dict)
        self.logger.info(f"BathymetrySource: Create source({time.time() - start:.1f}s)")

    # Realizing that Datasource is highly time dependent and doesn't make any sense for static data :/

    def instantiate_source_from_dict(self, source_dict: dict):
        if source_dict["source"] == "gebco":
            self.DataArray = self.get_bathymetry_from_file()
            self.grid_dict = self.get_grid_dict_from_xr(self.DataArray)
            print("instantiated source")
        else:
            raise NotImplementedError(
                f"Selected source {source_dict['source']} in the BathymetrySource dict is not implemented."
            )

    def get_bathymetry_from_file(self) -> xr:
        DataArray = xr.open_dataset(self.source_dict["source_settings"]["filepath"])
        # DataArray = DataArray.rename({"latitude": "lat", "longitude": "lon"})
        return DataArray

    def get_data_at_point(self, spatial_point: SpatialPoint) -> float:
        # TODO Watch out for mixin in DataSource (not used atm), compared to ocean datasource
        # TODO: Watch for order lon, lat
        return self.elevation_func(spatial_point.__array__())

    # TODO: why in comment y_grid, x_grid? Because the spatial point is lon, lat?
    # But spatiotemporal is [t, lat, lon] and
    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        # Note: the input to the casadi function needs to be an array of the form np.array([lat, lon])
        Args:
          grid:     list of the 2 grids [y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """

        self.elevation_func = ca.interpolant(
            "elevation", "linear", grid, array["elevation"].values.ravel(order="F")
        )

    # TODO: adapt
    def update_casadi_dynamics(self, state: PlatformState) -> None:
        """Function to update casadi_dynamics which means we fit an interpolant to grid data.
        Note: this can be overwritten in child-classes e.g. when an analytical function is available.
        Args:
          state: Platform State object containing [x, y, battery, mass, time] to update around
        """

        # Step 1: Create the intervals to query data for
        y_interval, x_interval, = self.convert_to_x_y_bounds(
            x_0=state.to_spatial_point(),
            x_T=state.to_spatial_point(),
            deg_around_x0_xT_box=self.source_dict["casadi_cache_settings"]["deg_around_x_t"],
            # temp_horizon_in_s=self.source_dict["casadi_cache_settings"]["time_around_x_t"],
        )

        # Step 2: Get the data from itself and update casadi_grid_dict
        xarray = self.get_data_over_area(x_interval, y_interval)
        self.casadi_grid_dict = self.get_grid_dict_from_xr(xarray)

        # Step 3: Set up the grid
        grid = [
            xarray.coords["lat"].values,
            xarray.coords["lon"].values,
        ]

        self.initialize_casadi_functions(grid, xarray)

    def get_data_over_area(
        self,
        x_interval: List[float],
        y_interval: List[float],
        spatial_resolution: Optional[float] = None,
    ) -> xr:
        """Function to get the the raw current data over an x, y interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          spatial_resolution: spatial resolution in the same units as x and y interval
        Returns:
          data_array     in xarray format that contains the grid and the values
        """
        # TODO: update docstring

        # Step 1: Subset the xarray accordingly
        # Step 1.1 include buffers around of the grid granularity (assuming regular grid axis)
        dlon = self.DataArray["lon"][1] - self.DataArray["lon"][0]
        x_interval_extended = [x_interval[0] - 1.5 * dlon.item(), x_interval[1] + 1.5 * dlon.item()]
        dlat = self.DataArray["lat"][1] - self.DataArray["lat"][0]
        y_interval_extended = [y_interval[0] - 1.5 * dlat.item(), y_interval[1] + 1.5 * dlat.item()]
        subset = self.DataArray.sel(
            lon=slice(x_interval_extended[0], x_interval_extended[1]),
            lat=slice(y_interval_extended[0], y_interval_extended[1]),
        )

        # Step 3: Do a sanity check for the sub-setting before it's used outside and leads to errors
        self.array_subsetting_sanity_check(subset, x_interval, y_interval, self.logger)

        # Step 4: perform interpolation to a specific resolution if requested
        if spatial_resolution is not None:
            subset = self.interpolate_in_space(subset, spatial_resolution)

        return subset

    @staticmethod
    def interpolate_in_space(array: xr, spatial_resolution: Optional[float]) -> xr:
        """Helper function for spatial interpolation"""

        # Run spatial interpolation
        if spatial_resolution is not None:
            lat_grid = np.arange(
                start=array["lat"][0].data, stop=array["lat"][-1].data, step=spatial_resolution
            )
            lon_grid = np.arange(
                start=array["lon"][0].data, stop=array["lon"][-1].data, step=spatial_resolution
            )
            array = array.interp(lon=lon_grid, lat=lat_grid, method="linear")

        return array

    @staticmethod
    def get_grid_dict_from_xr(xrDF: xr, source: Optional[AnyStr] = None) -> dict:
        """Helper function to extract the grid dict from an xrarray"""

        grid_dict = {
            "y_range": [xrDF["lat"].data[0], xrDF["lat"].data[-1]],
            "x_range": [xrDF["lon"].data[0], xrDF["lon"].data[-1]],
            "spatial_res": xrDF["lat"].data[1] - xrDF["lat"].data[0],
        }
        # TODO: do we need/want to extract more data?
        # if source != "opendap" and "water_u" in xrDF.variables:
        #     # If xrDF is not opendap, extract additional information (otherwise too big)
        #     grid_dict["y_grid"] = xrDF["lat"].data
        #     grid_dict["x_grid"] = xrDF["lon"].data
        #     grid_dict["t_grid"] = [
        #         units.get_posix_time_from_np64(np64) for np64 in xrDF["time"].data
        #     ]
        #     grid_dict["spatial_land_mask"] = np.ma.masked_invalid(
        #         xrDF.variables["water_u"].data[0, :, :]
        #     ).mask

        return grid_dict

    @staticmethod
    def array_subsetting_sanity_check(
        array: xr,
        x_interval: List[float],
        y_interval: List[float],
        logger: logging.Logger,
    ):
        """Advanced Check if admissible subset and warning of partially being out of bound in space."""
        # Step 1: collateral check is any dimension 0?
        if 0 in (len(array.coords["lat"]), len(array.coords["lon"])):
            raise ValueError("None of the requested spatial area is in the file.")
        # Step 2: Data partially not in the array check
        if (
            array.coords["lat"].data[0] > y_interval[0]
            or array.coords["lat"].data[-1] < y_interval[1]
        ):
            raise Exception(
                f"Part of the y requested area is outside of file(file: [{array.coords['lat'].data[0]}, {array.coords['lat'].data[-1]}], requested: [{y_interval[0]}, {y_interval[1]}])."
            )
        if (
            array.coords["lon"].data[0] > x_interval[0]
            or array.coords["lon"].data[-1] < x_interval[1]
        ):
            raise Exception(
                f"Part of the x requested area is outside of file (file: [{array.coords['lon'].data[0]}, {array.coords['lon'].data[-1]}], requested: [{x_interval[0]}, {x_interval[1]}])."
            )

    @staticmethod
    def convert_to_x_y_bounds(
        x_0: SpatialPoint,
        x_T: SpatialPoint,
        deg_around_x0_xT_box: float,
    ):
        """Helper function for spatial subsetting
        Args:
            x_0: SpatialPoint
            x_T: SpatialPoint goal locations
            deg_around_x0_xT_box: buffer around the box in degree

        Returns:
            lat_bnds: [y_lower, y_upper] in degrees
            lon_bnds: [x_lower, x_upper] in degrees
        """
        lon_bnds = [
            min(x_0.lon.deg, x_T.lon.deg) - deg_around_x0_xT_box,
            max(x_0.lon.deg, x_T.lon.deg) + deg_around_x0_xT_box,
        ]
        lat_bnds = [
            min(x_0.lat.deg, x_T.lat.deg) - deg_around_x0_xT_box,
            max(x_0.lat.deg, x_T.lat.deg) + deg_around_x0_xT_box,
        ]

        return lat_bnds, lon_bnds

    def map_analytical_function_over_area(self, grids_dict: dict) -> np.array:
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
        Args:
          grids_dict: containing ranges and grids of x, y, t dimension
        Returns:
          data_tuple     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        pass

    # def plot_data_over_area(self):
    #     pass

    # # TODO: probably we could do this with geopy for better accuracy
    # def is_on_land(self, point: SpatialPoint):
    #     """Helper function to check if a SpatialPoint is on the land indicated in the
    #         nc files as NaN (only approximate land boundaries).
    #         Accuracy is limited by the resolution of self.grid_dict.
    #     Args:
    #         point:    SpatialPoint object where to check if it is on land
    #     Returns:
    #         bool:     True if on land and false otherwise
    #     """
    #     if not self.grid_dict["x_grid"].min() < point.lon.deg < self.grid_dict["x_grid"].max():
    #         raise ValueError(
    #             f'Point {point} is not inside x_dict {self.grid_dict["x_grid"][[0,-1]]}'
    #         )
    #     if not self.grid_dict["y_grid"].min() < point.lat.deg < self.grid_dict["y_grid"].max():
    #         raise ValueError(
    #             f'Point {point} is not inside y_grid {self.grid_dict["y_grid"][[0,-1]]}'
    #         )

    #     x_idx = (np.abs(self.grid_dict["x_grid"] - point.lon.deg)).argmin()
    #     y_idx = (np.abs(self.grid_dict["y_grid"] - point.lat.deg)).argmin()
    #     return self.grid_dict["spatial_land_mask"][y_idx, x_idx]

    # # TODO: probably we could do this with geopy for better accuracy
    # def distance_to_land(self, point: SpatialPoint) -> units.Distance:
    #     """
    #         Helper function to get the distance of a SpatialPoint to land.
    #         Accuracy is limited by the resolution of self.grid_dict.
    #     Args:
    #         point:    SpatialPoint object where to calculate distance to land
    #     Returns:
    #         units.Distance:     Distance to closest land
    #     """
    #     if not self.grid_dict["x_grid"].min() < point.lon.deg < self.grid_dict["x_grid"].max():
    #         raise ValueError(
    #             f'Point {point} is not inside x_dict {self.grid_dict["x_grid"][[0,-1]]}'
    #         )
    #     if not self.grid_dict["y_grid"].min() < point.lat.deg < self.grid_dict["y_grid"].max():
    #         raise ValueError(
    #             f'Point {point} is not inside y_grid {self.grid_dict["y_grid"][[0,-1]]}'
    #         )

    #     lon1, lat1 = np.meshgrid(
    #         point.lon.deg * np.ones_like(self.grid_dict["x_grid"]),
    #         point.lat.deg * np.ones_like(self.grid_dict["y_grid"]),
    #     )
    #     lon2, lat2 = np.meshgrid(self.grid_dict["x_grid"], self.grid_dict["y_grid"])

    #     distances = np.vectorize(units.haversine_rad_from_deg)(lon1, lat1, lon2, lat2)
    #     land_distances = np.where(self.grid_dict["spatial_land_mask"], distances, np.inf)

    #     return units.Distance(rad=land_distances.min())

    def __del__(self):
        """Helper function to delete the existing casadi functions."""
        # print('__del__ called in OceanCurrentSource')
        del self.elevation_func
        pass
