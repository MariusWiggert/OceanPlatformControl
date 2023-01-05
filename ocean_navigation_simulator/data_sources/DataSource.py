"""The abstract base class for all Data Sources. Implements a lot of shared functionality"""
import abc
import datetime
import logging
import os
from functools import partial
from typing import Any, AnyStr, Callable, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import HTML
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.utils import units


class DataSourceException(Exception):
    pass


class SubsettingDataSourceException(DataSourceException):
    pass


class DataSource(abc.ABC):
    """Base class for various data sources."""

    logger: logging.Logger = logging.getLogger("data_source")

    @abc.abstractmethod
    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """

    def check_for_casadi_dynamics_update(self, state: PlatformState) -> bool:
        """Function to check if our cached casadi dynamics need an update because x_t is outside of the area.
        Args:
            state: Platform State to check if we have a working casadi function [x, y, battery, mass, posix_time]
            logger: logging object
        """
        out_x_range = not (
            self.casadi_grid_dict["x_range"][0]
            < state.lon.deg
            < self.casadi_grid_dict["x_range"][1]
        )
        out_y_range = not (
            self.casadi_grid_dict["y_range"][0]
            < state.lat.deg
            < self.casadi_grid_dict["y_range"][1]
        )
        out_t_range = not (
            self.casadi_grid_dict["t_range"][0]
            <= state.date_time
            < self.casadi_grid_dict["t_range"][1]
        )

        if out_x_range or out_y_range or out_t_range:
            if out_x_range:
                self.logger.debug(
                    f'Updating Interpolation (X: {self.casadi_grid_dict["x_range"][0]}, {state.lon.deg}, {self.casadi_grid_dict["x_range"][1]}'
                )
            if out_y_range:
                self.logger.debug(
                    f'Updating Interpolation (Y: {self.casadi_grid_dict["y_range"][0]}, {state.lat.deg}, {self.casadi_grid_dict["y_range"][1]}'
                )
            if out_t_range:
                self.logger.debug(
                    f'Updating Interpolation (T: {self.casadi_grid_dict["t_range"][0]}, {state.date_time}, {self.casadi_grid_dict["t_range"][1]}'
                )

            self.update_casadi_dynamics(state)
            return True
        return False

    def update_casadi_dynamics(self, state: PlatformState) -> None:
        """Function to update casadi_dynamics which means we fit an interpolant to grid data.
        Note: this can be overwritten in child-classes e.g. when an analytical function is available.
        Args:
          state: Platform State object containing [x, y, battery, mass, time] to update around
        """

        # Step 1: Create the intervals to query data for
        t_interval, y_interval, x_interval, = self.convert_to_x_y_time_bounds(
            x_0=state.to_spatio_temporal_point(),
            x_T=state.to_spatial_point(),
            deg_around_x0_xT_box=self.source_config_dict["casadi_cache_settings"].get(
                "deg_around_x_t", 0
            ),
            temp_horizon_in_s=self.source_config_dict["casadi_cache_settings"].get(
                "time_around_x_t", 0
            ),
        )

        # Step 2: Get the data from itself and update casadi_grid_dict
        xarray = self.get_data_over_area(x_interval, y_interval, t_interval, throw_exceptions=False)
        self.casadi_grid_dict = self.get_grid_dict_from_xr(xarray)

        # Step 3: Set up the grid
        grid = [
            units.get_posix_time_from_np64(xarray.coords["time"].values),
            xarray.coords["lat"].values,
            xarray.coords["lon"].values,
        ]

        self.initialize_casadi_functions(grid, xarray)

    @staticmethod
    def enforce_utc_datetime_object(time: Union[datetime.datetime, int]):
        """Takes a datetime object or posix timestamp and makes it timezone-aware by setting it to be in UTC."""
        # format to datetime object
        if not isinstance(time, datetime.datetime):
            return datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)
        elif time.tzinfo is None:
            return time.replace(tzinfo=datetime.timezone.utc)
        return time

    @staticmethod
    def convert_to_x_y_time_bounds(
        x_0: SpatioTemporalPoint,
        x_T: SpatialPoint,
        deg_around_x0_xT_box: float,
        temp_horizon_in_s: float,
    ):
        """Helper function for spatio-temporal subsetting
        Args:
            x_0: SpatioTemporalPoint
            x_T: SpatialPoint goal locations
            deg_around_x0_xT_box: buffer around the box in degree
            temp_horizon_in_s: maximum temp_horizon to look ahead of x_0 time in seconds

        Returns:
            t_interval: if time-varying: [t_0, t_T] as utc datetime objects
                        where t_0 and t_T are the start and end respectively
            lat_bnds: [y_lower, y_upper] in degrees
            lon_bnds: [x_lower, x_upper] in degrees
        """
        t_interval = [x_0.date_time, x_0.date_time + datetime.timedelta(seconds=temp_horizon_in_s)]
        # somehow jax DeviceArray ended up here, float solves it for the time beeing.
        lon_bnds = [
            float(min(x_0.lon.deg, x_T.lon.deg) - deg_around_x0_xT_box),
            float(max(x_0.lon.deg, x_T.lon.deg) + deg_around_x0_xT_box),
        ]
        lat_bnds = [
            float(min(x_0.lat.deg, x_T.lat.deg) - deg_around_x0_xT_box),
            float(max(x_0.lat.deg, x_T.lat.deg) + deg_around_x0_xT_box),
        ]

        return t_interval, lat_bnds, lon_bnds

    @staticmethod
    def get_grid_dict_from_xr(xrDF: xr, source: Optional[AnyStr] = None) -> dict:
        """Helper function to extract the grid dict from an xrarray"""

        grid_dict = {
            "t_range": [
                units.get_datetime_from_np64(np64)
                for np64 in [xrDF["time"].data[0], xrDF["time"].data[-1]]
            ],
            "y_range": [xrDF["lat"].data[0], xrDF["lat"].data[-1]],
            "x_range": [xrDF["lon"].data[0], xrDF["lon"].data[-1]],
            "spatial_res": xrDF["lat"].data[1] - xrDF["lat"].data[0],
            "temporal_res": (xrDF["time"].data[1] - xrDF["time"].data[0]) / np.timedelta64(1, "s"),
        }
        if source != "opendap" and "water_u" in xrDF.variables:
            # If xrDF is not opendap, extract additional information (otherwise too big)
            grid_dict["y_grid"] = xrDF["lat"].data
            grid_dict["x_grid"] = xrDF["lon"].data
            grid_dict["t_grid"] = [
                units.get_posix_time_from_np64(np64) for np64 in xrDF["time"].data
            ]
            grid_dict["spatial_land_mask"] = np.ma.masked_invalid(
                xrDF.variables["water_u"].data[0, :, :]
            ).mask

        return grid_dict

    def array_subsetting_sanity_check(
        self,
        array: xr,
        x_interval: List[float],
        y_interval: List[float],
        t_interval: List[datetime.datetime],
        logger: logging.Logger,
        throw_exception: Optional[bool] = True,
    ):
        """Advanced Check if admissible subset and warning of partially being out of bound in space or time."""
        temporal_error = False
        spatial_error = False

        # Step 1: collateral check is any dimension 0?
        if 0 in (len(array.coords["lat"]), len(array.coords["lon"]), len(array.coords["time"])):
            # check which dimension for more informative errors
            if len(array.coords["time"]) == 0:
                temporal_error = "None of the requested t_interval is in the file."
            else:
                spatial_error = "None of the requested spatial area is in the file."

        # Step 2: Data partially not in the array check
        if (
            array.coords["lat"].data[0] >= y_interval[0]
            or array.coords["lat"].data[-1] <= y_interval[1]
        ):
            spatial_error = f"Part of the y requested area is outside of file (requested: [{y_interval[0]}, {y_interval[1]}])."
        if (
            array.coords["lon"].data[0] >= x_interval[0]
            or array.coords["lon"].data[-1] <= x_interval[1]
        ):
            spatial_error = f"Part of the x requested area is outside of file (requested: [{x_interval[0]}, {x_interval[1]}])."
        if units.get_datetime_from_np64(array.coords["time"].data[0]) > t_interval[0]:
            temporal_error = f"The starting time is not in the array (requested: [{t_interval[0]}, {t_interval[1]}])."
        if units.get_datetime_from_np64(array.coords["time"].data[-1]) < t_interval[1]:
            temporal_error = f"The requested final time is not part of the subset (requested: [{t_interval[0]}, {t_interval[1]}])."

        if temporal_error or spatial_error:
            error = temporal_error or spatial_error
            error += " (files: x_range: {x}, y_range: {y}, t_range:{t})".format(
                x=self.grid_dict["x_range"],
                y=self.grid_dict["y_range"],
                t=[
                    self.grid_dict["t_range"][0].strftime("%Y-%m-%d %H-%M-%S"),
                    self.grid_dict["t_range"][-1].strftime("%Y-%m-%d %H-%M-%S"),
                ],
            )
            if throw_exception and ():
                raise SubsettingDataSourceException(error)
            elif temporal_error or spatial_error:
                logger.warning(error)

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

        # format to datetime object
        time = self.enforce_utc_datetime_object(time=time)

        # Step 1: get the area data
        area_xarray = self.get_data_over_area(
            x_interval,
            y_interval,
            [time, time + datetime.timedelta(seconds=1)],
            spatial_resolution=spatial_resolution,
        )

        # interpolate to specific time
        atTimeArray = area_xarray.interp(time=time.replace(tzinfo=None))

        # Plot the current field
        ax = self.plot_xarray_for_animation(time_idx=0, xarray=atTimeArray, ax=ax, **kwargs)
        if return_ax:
            return ax
        else:
            plt.show()

    @staticmethod
    def set_up_geographic_ax() -> matplotlib.pyplot.axes:
        """Helper function to set up a geographic ax object to plot on."""
        ax = plt.axes(projection=ccrs.PlateCarree())
        grid_lines = ax.gridlines(draw_labels=True, zorder=4, color="silver", alpha=1)
        grid_lines.top_labels = False
        grid_lines.right_labels = False
        ax.add_feature(cfeature.LAND, zorder=3, edgecolor="black")
        return ax

    def bound_spatial_temporal_resolution(
        self,
        x_interval: List[float],
        y_interval: List[float],
        max_spatial_n: Optional[int] = None,
        max_temp_n: Optional[int] = None,
    ) -> Tuple[Union[float, Any], Any]:
        """Helper Function to upper bound the resolutions for plotting.
        Args:
            x_interval:       List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
            y_interval:       List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
            max_spatial_n:    Per default the data_source resolution is used. If that is too much, downscaled to this.
            max_temp_n:       Per default the data_source temp resolution is used. If too many, we downscale it.
        Returns:
            spatial_res:       None or adjusted spatial resolution
            temporal_res:      None or adjusted spatial resolution
        """
        # Per default we leave it as None to minimize compute
        spatial_res, temporal_res = [None] * 2
        # Check if spatial sub-setting would give more than max_spatial_n
        if max_spatial_n is not None:
            n_x = (x_interval[1] - x_interval[0]) / self.grid_dict["spatial_res"]
            n_y = (y_interval[1] - y_interval[0]) / self.grid_dict["spatial_res"]
            max_data_n = max(n_y, n_x)
            # adjust spatial resolution
            if max_data_n > max_spatial_n:
                spatial_res = (max_data_n / max_spatial_n) * self.grid_dict["spatial_res"]

        # Check if temporal sub-setting would give more than max_temp_n
        if max_temp_n is not None:
            n_t = (x_interval[1] - x_interval[0]) / self.grid_dict["temporal_res"]
            # adjust temporal resolution
            if n_t > max_temp_n:
                spatial_res = (n_t / max_temp_n) * self.grid_dict["temporal_res"]

        return spatial_res, temporal_res

    @staticmethod
    def render_animation(animation_object: Any, fps: int = 10, output: AnyStr = None):
        """Helper Function to render animations once created.
        Args:
            animation_object:       Matplotlib Animation object
            fps:                    Frames per Second
            output:                 How to output the animation. Options are either saved to file or via html in jupyter/safari.
                                    Strings in {'*.mp4', '*.gif', 'safari', 'jupyter'}
        """
        folder_to_save_in = "generated_media/" if not output.startswith("/") else ""

        # Now render it to a file
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if output in ["safari", "jupyter"]:
                ani_html = HTML(animation_object.to_html5_video())
                # Render using safari (only used because of Pycharm local testing)
                if output == "safari":
                    with open(folder_to_save_in + "data_animation.html", "w") as file:
                        file.write(ani_html.data)
                        os.system(
                            'open "/Applications/Safari.app" '
                            + '"'
                            + os.path.realpath(folder_to_save_in + "data_animation.html")
                            + '"'
                        )
                else:  # visualize in Jupyter directly
                    plt.close()
                    return ani_html
            elif ".gif" in output:
                animation_object.save(
                    folder_to_save_in + output, writer=animation.PillowWriter(fps=fps)
                )
                plt.close()
            elif ".mp4" in output:
                animation_object.save(
                    folder_to_save_in + output, writer=animation.FFMpegWriter(fps=fps)
                )
                plt.close()
            else:
                raise ValueError(
                    "save_as_filename can be either None (for HTML rendering) or filepath and name needs to"
                    "contain either '.gif' or '.mp4' to specify the format and desired file location."
                )

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
            if self.source_config_dict["use_geographic_coordinate_system"]:
                ax = self.set_up_geographic_ax()
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
        colorbar: bool = True,
        return_cbar: bool = False,
        set_title: bool = True,
        **kwargs,
    ) -> Union[matplotlib.pyplot.axes, Tuple[matplotlib.pyplot.axes, Figure.colorbar]]:
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
        if var_to_plot is None:
            var_to_plot = list(xarray.keys())[0]

        # Step 1: Make the data ready for plotting
        # check if time-dimension already collapsed or not yet
        if xarray["time"].size != 1:
            xarray = xarray.isel(time=time_idx)
        time = units.get_datetime_from_np64(xarray["time"].data)

        if ax is None:
            ax = plt.axes()

        # plot data for the specific variable
        if vmax is None:
            vmax = xarray[var_to_plot].max()
        if vmin is None:
            vmin = xarray[var_to_plot].min()
        im = xarray[var_to_plot].plot(
            cmap="viridis", vmin=vmin, vmax=vmax, alpha=alpha, ax=ax, add_colorbar=False
        )

        # set and format colorbar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(position="right", size="5%", pad=0.15, axes_class=plt.Axes)
            cbar = plt.colorbar(im, orientation="vertical", cax=cax)
            cbar.ax.set_ylabel("current velocity in m/s")
            cbar.set_ticks(cbar.get_ticks())
            precision = 1
            if int(vmin * 10) == int(vmax * 10):
                precision = 2 if int(vmin * 100) != int(vmin * 100) else 3
            cbar.set_ticklabels(
                ["{:.{prec}f}".format(elem, prec=precision) for elem in cbar.get_ticks().tolist()]
            )

        # Label the plot
        if set_title:
            ax.set_title(
                "Variable: {var} \n at Time: {t}".format(
                    var=var_to_plot, t="Time: " + time.strftime("%Y-%m-%d %H:%M:%S UTC")
                )
            )
        if return_cbar:
            return ax, cbar

        return ax

    def animate_data(
        self,
        x_interval: List[float],
        y_interval: List[float],
        t_interval: List[Union[datetime.datetime, float]],
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
        add_ax_func: Optional[Callable] = None,
        fps: int = 10,
        output: AnyStr = "data_animation.mp4",
        forward_time: bool = True,
        **kwargs,
    ):
        """Basis function to animate data over a specific area and time interval.
        Args:
          x_interval:       List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval:       List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval:       List of the lower and upper time either datetime.datetime or posix.
          spatial_resolution:   Per default (None) the data_source resolution is used otherwise the selected one.
          temporal_resolution:  Per default (None) the data_source temp resolution is used otherwise the selected one.
          add_ax_func:      function handle what to add on top of the current visualization
                            signature needs to be such that it takes an axis object and time as input
                            e.g. def add(ax, time, x=10, y=4): ax.scatter(x,y) always adds a point at (10, 4)
          var_to_plot:      The variable to be plotted. If None default settings for the field is used.
          fps:              Frames per second
          output:           How to output the animation. Options are either saved to file or via html in jupyter/safari.
                            Strings in {'*.mp4', '*.gif', 'safari', 'jupyter'}
          forward_time:     If True, animation is forward in time, if false backwards
          **kwargs:         Further keyword arguments for plotting(see plot_currents_from_xarray)
        """

        # Step 1: get the data_subset for animation
        get_data_dict = {
            "x_interval": x_interval,
            "y_interval": y_interval,
            "t_interval": t_interval,
            "spatial_resolution": spatial_resolution,
            "temporal_resolution": temporal_resolution,
        }
        # If we only want to animate noise, add this to the get_data_dict
        if "noise_only" in kwargs:
            get_data_dict.update({"noise_only": kwargs["noise_only"]})
            del kwargs["noise_only"]

        xarray = self.get_data_over_area(**get_data_dict)

        # Calculate min and max over the full tempo-spatial array
        if self.source_config_dict["field"] == "OceanCurrents":
            # get rounded up vmax across the whole dataset (with ` decimals)
            xarray = xarray.assign(magnitude=lambda x: (x.water_u**2 + x.water_v**2) ** 0.5)
            vmax = round(xarray["magnitude"].max().item() + 0.049, 1)
            vmin = 0
        else:
            vmax = units.round_to_sig_digits(
                xarray[list(xarray.keys())[0]].max().item(), sig_digit=2
            )
            vmin = units.round_to_sig_digits(
                xarray[list(xarray.keys())[0]].min().item(), sig_digit=2
            )

        # add vmin and vmax to kwargs if not already in it
        if "vmax" not in kwargs:
            kwargs.update({"vmax": vmax})
        if "vmin" not in kwargs:
            kwargs.update({"vmin": vmin})

        # create global figure object where the animation happens
        if "figsize" in kwargs:
            fig = plt.figure(figsize=kwargs["figsize"])
        else:
            fig = plt.figure(figsize=(12, 12))

        # create a partial function with most variables already set for the animation loop to call
        if add_ax_func is not None:
            # define full function for rendering the frame
            def full_plot_func(time_idx):
                # plot underlying currents at time
                ax = self.plot_xarray_for_animation(
                    time_idx=time_idx,
                    xarray=xarray,
                    reset_plot=True,
                    **kwargs,
                )
                # extract the posix time
                posix_time = units.get_posix_time_from_np64(xarray["time"].data[time_idx])
                # add the ax_adding_func
                add_ax_func(ax, posix_time)

            # create partial func from the full_function
            render_frame = partial(full_plot_func)
        else:
            render_frame = partial(
                self.plot_xarray_for_animation,
                xarray=xarray,
                reset_plot=True,
                **kwargs,
            )

        # set time direction of the animation
        frames_vector = np.where(
            forward_time, np.arange(xarray["time"].size), np.flip(np.arange(xarray["time"].size))
        )
        # create animation function object (it's not yet executed)
        ani = animation.FuncAnimation(fig, func=render_frame, frames=frames_vector, repeat=False)

        # render the animation with the keyword arguments
        self.render_animation(animation_object=ani, output=output, fps=fps)

    def check_for_most_recent_fmrc_dataframe(self, time: datetime.datetime) -> datetime:
        """Helper function to check update the self.OceanCurrent if a new forecast is available at
        the specified input time.
        Args:
          time: datetime object
        Output:
           an integer representing the last file index on which it has planned
        """
        return 0


# Two types of data sources: analytical and xarray based ones -> need different default functions, used via mixin
class XarraySource(abc.ABC):
    def __init__(self, source_config_dict: dict):
        self.source_config_dict = source_config_dict
        self.DataArray = None  # The xarray containing the raw data (if not an analytical function)
        self.grid_dict, self.casadi_grid_dict = [None] * 2

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> xr:
        """Function to get the data at a specific point.
        Args:
          spatio_temporal_point: SpatioTemporalPoint in the respective used coordinate system geospherical or unitless
        Returns:
          xr object that is then processed by the respective data source for its purpose
        """

        return self.DataArray.interp(
            time=np.datetime64(spatio_temporal_point.date_time),
            lon=spatio_temporal_point.lon.deg,
            lat=spatio_temporal_point.lat.deg,
            method="linear",
        )

    def get_data_over_area(
        self,
        x_interval: List[float],
        y_interval: List[float],
        t_interval: List[Union[datetime.datetime, int]],
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
        throw_exceptions: Optional[bool] = True,
        spatial_tolerance: Optional[float] = 2.0,
    ) -> xr:
        """Function to get the the raw current data over an x, y, and t interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime or posix float
          spatial_resolution: spatial resolution in the same units as x and y interval
          temporal_resolution: temporal resolution in seconds
          throw_exceptions:    if True an exception is thrown when requesting data not available otherwise warning
          spatial_tolerance:   how much spatial grid sizes the interval is extended to make sure it is inside
        Returns:
          data_array     in xarray format that contains the grid and the values (land is NaN)
        """
        # format to datetime object
        if not isinstance(t_interval[0], datetime.datetime):
            t_interval = [
                datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)
                for time in t_interval
            ]
        # Step 1: Subset the xarray accordingly
        # Step 1.1 include buffers around of the grid granularity (assuming regular grid axis)
        dt = self.DataArray["time"][1] - self.DataArray["time"][0]
        t_interval_extended = [
            np.datetime64(t_interval[0].replace(tzinfo=None)) - dt,
            np.datetime64(t_interval[1].replace(tzinfo=None)) + dt,
        ]
        dlon = self.DataArray["lon"][1] - self.DataArray["lon"][0]
        x_interval_extended = [
            x_interval[0] - spatial_tolerance * dlon.item(),
            x_interval[1] + spatial_tolerance * dlon.item(),
        ]
        dlat = self.DataArray["lat"][1] - self.DataArray["lat"][0]
        y_interval_extended = [
            y_interval[0] - spatial_tolerance * dlat.item(),
            y_interval[1] + spatial_tolerance * dlat.item(),
        ]
        subset = self.DataArray.sel(
            time=slice(t_interval_extended[0], t_interval_extended[1]),
            lon=slice(x_interval_extended[0], x_interval_extended[1]),
            lat=slice(y_interval_extended[0], y_interval_extended[1]),
        )

        # Step 3: Do a sanity check for the sub-setting before it's used outside and leads to errors
        self.array_subsetting_sanity_check(
            subset, x_interval, y_interval, t_interval, self.logger, throw_exceptions
        )

        # Step 4: perform interpolation to a specific resolution if requested
        if spatial_resolution is not None or temporal_resolution is not None:
            subset = self.interpolate_in_space_and_time(
                subset, spatial_resolution, temporal_resolution
            )

        return subset

    @staticmethod
    def interpolate_in_space_and_time(
        array: xr, spatial_resolution: Optional[float], temporal_resolution: Optional[float]
    ) -> xr:
        """Helper function for temporal and spatial interpolation"""
        # Run temporal interpolation
        if temporal_resolution is not None:
            time_grid = np.arange(
                start=array["time"][0].data,
                stop=array["time"][-1].data,
                step=np.timedelta64(temporal_resolution, "s"),
            )
            array = array.interp(time=time_grid, method="linear")

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


class AnalyticalSource(abc.ABC):
    def __init__(self, source_config_dict: dict):
        """Class for Analytical Ocean Current Sources
        Args:
          source_config_dict: dict the key 'source_settings' to a dict with the relevant specific settings
            The general AnalyticalSource requires the following keys, the explicit analytical currents some extra.
                spatial_domain:
                        a list e..g [np.array([-0.1, -0.1]), np.array([2.1, 1.1])],
                temporal_domain:
                        a list e.g. [-10, 1000] of the temporal domain in units (will internally be seconds)
                spatial_resolution:
                        a float as the default spatial_resolution in degree
                temporal_resolution:
                        a float as the default temporal_resolution in seconds
                boundary_buffers (Optional, Default is [0, 0]
                        Margin to buffer the spatial domain with obstacles as boundary conditions e.g. [0.2, 0.2]
                        Note: this is only required for OceanCurrent Analytical Sources that HJ Reachability runs stable.
        """
        self.source_config_dict = source_config_dict
        self.grid_dict, self.casadi_grid_dict = [None] * 2

        # Step 1: Some basic initializations
        # adjust spatial domain by boundary buffer
        if "boundary_buffers" not in source_config_dict["source_settings"]:
            self.spatial_boundary_buffers = np.array([0.0, 0.0])
        else:
            self.spatial_boundary_buffers = np.array(
                source_config_dict["source_settings"]["boundary_buffers"]
            )
        self.x_domain = [
            x + self.spatial_boundary_buffers[0] * (-1) ** i
            for x, i in zip(source_config_dict["source_settings"]["x_domain"], [1, 2])
        ]
        self.y_domain = [
            y + self.spatial_boundary_buffers[1] * (-1) ** i
            for y, i in zip(source_config_dict["source_settings"]["y_domain"], [1, 2])
        ]
        # set the temp_domain_posix
        if isinstance(
            source_config_dict["source_settings"]["temporal_domain"][0], datetime.datetime
        ):
            # Assume it's utc if not correct it
            self.temp_domain_posix = [
                t.replace(tzinfo=datetime.timezone.utc).timestamp()
                for t in source_config_dict["source_settings"]["temporal_domain"]
            ]
        else:
            self.temp_domain_posix = source_config_dict["source_settings"]["temporal_domain"]
        # Set the default resolutions (used when none is provided in get_data_over_area)
        self.spatial_resolution = source_config_dict["source_settings"]["spatial_resolution"]
        self.temporal_resolution = source_config_dict["source_settings"]["temporal_resolution"]

        # Step 3: derive a general grid_dict
        self.grid_dict = self.get_ranges_dict()

    @abc.abstractmethod
    def create_xarray(self, grids_dict: dict, data_tuple: Tuple) -> xr:
        """Function to create an xarray from the data tuple and grid dict
        Args:
          data_tuple: tuple containing the data of the source as numpy array
          grids_dict: containing ranges and grids of x, y, t dimension
        Returns:
          xr     an xarray containing both the grid and data
        """

    @abc.abstractmethod
    def map_analytical_function_over_area(self, grids_dict: dict):
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
        Args:
          grids_dict: containing grids of x, y, t dimension
        Returns:
          data     containing the data in whatever format as numpy array (not yet in xarray form) e.g. Tuple
        """

    @abc.abstractmethod
    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> xr:
        """Function to get the data at a specific point.
        Args:
          spatio_temporal_point: SpatioTemporalPoint in the respective used coordinate system geospherical or unitless
        Returns:
          xr object that is then processed by the respective data source for its purpose
        """
        raise NotImplementedError

    def get_data_over_area(
        self,
        x_interval: List[float],
        y_interval: List[float],
        t_interval: List[Union[datetime.datetime, float]],
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
    ) -> xr:
        """Function to get the the raw current data over an x, y, and t interval.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime or posix.
          spatial_resolution: spatial resolution in the same units as x and y interval
          temporal_resolution: temporal resolution in seconds
        Returns:
          data_array     in xarray format that contains the grid and the values (land is NaN)
        """

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

        return subset

    def get_ranges_dict(
        self,
        x_interval: Optional[List[float]] = None,
        y_interval: Optional[List[float]] = None,
        t_interval: Optional[List[float]] = None,
    ):
        """Helper function to get a ranges dictionary bounded by the spatial and temporal domain.
        If no input is provided this returns the ranges for data source domain.
        Args:
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          t_interval: List of the lower and upper datetime requested [t_0, t_T] in POSIX time
        """

        # Step 1: Check default interval or bounded by the respective domain of the Data Source
        if t_interval is None:
            t_interval = self.temp_domain_posix
        else:
            t_interval = [
                max(t_interval[0], self.temp_domain_posix[0]),
                min(t_interval[1], self.temp_domain_posix[1]),
            ]
        if x_interval is None:
            x_interval = [self.x_domain[0], self.x_domain[1]]
        else:
            x_interval = [
                max(x_interval[0], self.x_domain[0]),
                min(x_interval[1], self.x_domain[1]),
            ]
        if y_interval is None:
            y_interval = [self.y_domain[0], self.y_domain[1]]
        else:
            y_interval = [
                max(y_interval[0], self.y_domain[0]),
                min(y_interval[1], self.y_domain[1]),
            ]

        return {
            "y_range": y_interval,
            "x_range": x_interval,
            "t_range": [
                datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc) for t in t_interval
            ],
            "temporal_res": self.temporal_resolution,
            "spatial_res": self.spatial_resolution,
        }

    def get_grid_dict(
        self,
        x_interval: Optional[List[float]] = None,
        y_interval: Optional[List[float]] = None,
        t_interval: Optional[List[float]] = None,
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
    ):

        """Helper Function to produce a grid dict."""
        if spatial_resolution is None:
            spatial_resolution = self.spatial_resolution
        if temporal_resolution is None:
            temporal_resolution = self.temporal_resolution

        # Step 1: Get the domain adjusted range dict
        ranges_dict = self.get_ranges_dict(x_interval, y_interval, t_interval)

        # Step 2: Get the grids with the respective resolutions
        # Step 2.1 Spatial coordinate vectors with desired resolution
        # The + spatial_resolution is a hacky way to include the endpoint. We want a regular grid hence the floor to two decimals.
        x_vector = np.arange(
            start=np.floor(ranges_dict["x_range"][0] * 100) / 100,
            stop=ranges_dict["x_range"][1] + 1.1 * spatial_resolution,
            step=spatial_resolution,
        )
        y_vector = np.arange(
            start=np.floor(ranges_dict["y_range"][0] * 100) / 100,
            stop=ranges_dict["y_range"][1] + 1.1 * spatial_resolution,
            step=spatial_resolution,
        )

        # Step 2.2 Temporal grid in POSIX TIME. We want a regular grid hence the floor to two decimals.
        t_grid = np.arange(
            start=np.floor((t_interval[0] - temporal_resolution) * 100) / 100,
            stop=t_interval[1] + 1.1 * temporal_resolution,
            step=temporal_resolution,
        )

        return {"x_grid": x_vector, "y_grid": y_vector, "t_grid": t_grid}

    def is_boundary(
        self,
        lon: Union[float, np.array],
        lat: Union[float, np.array],
        posix_time: Union[float, np.array],
    ) -> Union[float, np.array]:
        """Helper function to check if a state is in the boundary specified in hj as obstacle."""
        x_boundary = np.logical_or(
            lon < self.x_domain[0] + self.spatial_boundary_buffers[0],
            lon > self.x_domain[1] - self.spatial_boundary_buffers[0],
        )
        y_boundary = np.logical_or(
            lat < self.y_domain[0] + self.spatial_boundary_buffers[1],
            lat > self.y_domain[1] - self.spatial_boundary_buffers[1],
        )

        return np.logical_or(x_boundary, y_boundary)
