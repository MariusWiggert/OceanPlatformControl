import datetime
import logging
import os
import time
from typing import AnyStr, Dict, List, Optional, Tuple, Union

import casadi as ca
import cmocean
import matplotlib.pyplot
import matplotlib.pyplot as plt
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
                          e.g. {'deg_around_x_t': 2}
            source_dict (Dict): _description_
            use_geographic_coordinate_system (Optional[bool], optional): _description_. Defaults to True.
        """
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

    def instantiate_source_from_dict(self, source_dict: dict):
        if source_dict["source"] == "gebco":
            self.DataArray = self.get_bathymetry_from_file()
            self.grid_dict = self.get_grid_dict_from_xr(self.DataArray)
        else:
            raise NotImplementedError(
                f"Selected source {source_dict['source']} in the BathymetrySource dict is not implemented."
            )

    def get_bathymetry_from_file(self) -> xr:
        DataArray = xr.open_dataset(self.source_dict["source_settings"]["filepath"])
        # DataArray = DataArray.rename({"latitude": "lat", "longitude": "lon"})
        return DataArray

    def get_data_at_point(self, spatial_point: SpatialPoint) -> float:
        # TODO: Watch for order lon, lat!
        return self.elevation_func(spatial_point.__array__()[::-1])

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
        return grid_dict

    @staticmethod
    def array_subsetting_sanity_check(
        array: xr,
        x_interval: List[float],
        y_interval: List[float],
        logger: logging.Logger,
    ):
        """Advanced Check if admissible subset and warning of partially being out of bound in space."""
        # Step 1: collateral check if any dimension is 0
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

    def is_higher_than(self, point: SpatialPoint, elevation: float = 0):
        """Helper function to check if a SpatialPoint is on the land.
            Accuracy is limited by the resolution of self.grid_dict.
        Args:
            point:    SpatialPoint object where to check if it is on land
        Returns:
            bool:     True if on land and false otherwise
        """

        if not (
            self.casadi_grid_dict["x_range"][0]
            < point.lon.deg
            < self.casadi_grid_dict["x_range"][1]
        ):
            raise ValueError(
                f"Point {point} is not in casadi_grid_dict lon range{self.casadi_grid_dict['x_range']}"
            )

        if not (
            self.casadi_grid_dict["y_range"][0]
            < point.lat.deg
            < self.casadi_grid_dict["y_range"][1]
        ):
            raise ValueError(
                f"Point {point} is not in casadi_grid_dict lat range {self.casadi_grid_dict['y_range']}"
            )

        return self.get_data_at_point(point) > elevation

    def plot_data_at_time_over_area(
        self,
        time: Union[datetime.datetime, float],
        x_interval: List[float],
        y_interval: List[float],
        spatial_resolution: Optional[float] = None,
        return_ax: Optional[bool] = False,
        ax: Optional[matplotlib.pyplot.axes] = None,
        **kwargs,
    ):
        """Plot the data at a specific time over an area defined by the x and y intervals.
        Args:
          time:             time for which to plot the data either posix or datetime.datetime object
          x_interval:       List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval:       List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          spatial_resolution:Per default (None) the data_source resolution is used otherwise the selected one.
          return_ax:        if True returns ax, otherwise renders plots with plt.show()
          ax:               input axis object to plot on top of
          **kwargs:         Further keyword arguments for more specific setting, see plot_currents_from_2d_xarray.
        """

        # Format to datetime object
        if not isinstance(time, datetime.datetime):
            time = datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)

        # Step 1: get the area data
        area_xarray = self.get_data_over_area(
            x_interval,
            y_interval,
            spatial_resolution=spatial_resolution,
        )

        # interpolate to specific time
        # atTimeArray = area_xarray.interp(time=time.replace(tzinfo=None))

        # Plot the current field
        ax = self.plot_xarray_for_animation(time_idx=0, xarray=area_xarray, ax=ax, **kwargs)
        if return_ax:
            return ax
        else:
            plt.show()

    def plot_xarray_for_animation(
        self,
        time_idx: int,
        xarray: xr,
        reset_plot: Optional[bool] = False,
        figsize: Tuple[int] = (6, 6),
        ax: matplotlib.pyplot.axes = None,
        **kwargs,
    ) -> matplotlib.pyplot.axes:
        """Helper function for animations adding plot resets, figure size and automatically generating the axis.
        See plot_data_from_xarray for other optional keyword arguments.
        Args:
            time_idx:          time-idx to select from the xarray (only if it has time dimension)
            xarray:            xarray object containing the grids and data
            reset_plot:        if True the current figure is re-setted otherwise a new figure created (used for animation)
            figsize:           figure size
            ax:                Optional an axis object as input to plot on top of.
        Returns:
            ax                 matplotlib.pyplot.axes object
        """

        # reset plot this is needed for matplotlib.animation
        if reset_plot:
            plt.clf()
        else:  # create a new figure object where this is plotted
            plt.figure(figsize=figsize)

        # Step 2: Create ax object
        if ax is None:
            if self.source_dict["use_geographic_coordinate_system"]:
                # TODO: fix later
                # ax = self.set_up_geographic_ax()
                ax = plt.axes()
            else:
                ax = plt.axes()

        return self.plot_data_from_xarray(time_idx=time_idx, xarray=xarray, ax=ax, **kwargs)

    @staticmethod
    def plot_data_from_xarray(
        time_idx: int,
        xarray: xr,
        var_to_plot: AnyStr = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: Optional[float] = 1.0,
        ax: plt.axes = None,
        fill_nan: bool = True,
    ) -> matplotlib.pyplot.axes:
        """Base function to plot a specific var_to_plot of the x_array. If xarray has a time-dimension time_idx is selected,
        if xarray's time dimension is already collapsed (e.g. after interpolation) it's directly plotted.
        All other functions build on top of it, it creates the ax object and returns it.
        Args:
            time_idx:          time-idx to select from the xarray (only if it has time dimension)
            xarray:            xarray object containing the grids and data
            var_to_plot:       a string of the variable to plot
            vmin:              minimum current magnitude used for colorbar (float)
            vmax:              maximum current magnitude used for colorbar (float)
            alpha:             alpha of the current magnitude color visualization
            ax:                Optional for feeding in an axis object to plot the figure on.
            fill_nan:          Optional if True we fill nan values with 0 otherwise leave them as nans.
        Returns:
            ax                 matplotlib.pyplot.axes object
        """
        if fill_nan:
            xarray = xarray.fillna(0)
        # Get data variable if not provided
        # TODO: fix
        if var_to_plot is None:
            var_to_plot = "elevation"
        if ax is None:
            ax = plt.axes()

        # plot data for the specific variable
        if vmax is None:
            vmax = xarray[var_to_plot].max()
        if vmin is None:
            vmin = xarray[var_to_plot].min()
        # TODO: adapt when separated into classes
        xarray[var_to_plot].plot(cmap="cmo.topo", vmin=-6000, vmax=6000, alpha=alpha, ax=ax)
        # Label the plot
        ax.set_title(
            "Variable: {var} \n at Time: {t}".format(
                var=var_to_plot, t="Time: " + time.strftime("%Y-%m-%d %H:%M:%S UTC")
            )
        )

        return ax

    def __del__(self):
        """Helper function to delete the existing casadi functions."""
        del self.elevation_func
        pass
