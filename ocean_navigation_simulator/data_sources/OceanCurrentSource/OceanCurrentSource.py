import datetime
import logging
import os
from typing import AnyStr, List, Optional, Union

import casadi as ca
import dask.array.core
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pydap.cas.get_cookies import setup_session
from pydap.client import open_url

from ocean_navigation_simulator.data_sources.DataSource import (
    DataSource,
    XarraySource,
)
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentVector import (
    OceanCurrentVector,
)
from ocean_navigation_simulator.environment.PlatformState import (
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.units import (
    get_datetime_from_np64,
    get_posix_time_from_np64,
)


# TODO: Ok to pass data with NaNs to check for out of bound with point data? Or fill with 0?
# The fill with 0 could also be done in the HJ Planner, then we don't need to save the land grid anywhere.
# => Land checking speed test with the mask in the data vs. with polygon checking
# => also check if we really need the x and y grid of the data anywhere, I don't think so. => only for is_on_land checking!
# => lets write the rest, but I have a hunch we can eliminate that soon.
# TODO: most likely we don't even need to maintain the grid_dict of the dataset or do we?
# currently no safety mechanism build in to check if daily files or forecasts are consecutive.
# https://stackoverflow.com/questions/68170708/counting-consecutive-days-of-temperature-data
# Via diff in pandas and checking if consistent
# TODO: NaN handling as obstacles in the HJ planner would be really useful! (especially for analytical!)
# TODO: Does not work well yet for getting the most recent forecast point data (because needs to load casadi in)
# TODO: Clean up the mess with get_data_at_point, maybe by checking if caching function exists or not?
# Note: Error files from generated noise from jonas reading is a hack right now, need to change that!


class OceanCurrentSource(DataSource):
    """Base class for various data sources of ocean currents to handle different current sources."""

    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        # Note: the input to the casadi function needs to be an array of the form np.array([posix time, lat, lon])
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """

        self.u_curr_func = ca.interpolant(
            "u_curr", "linear", grid, array["water_u"].values.ravel(order="F")
        )
        self.v_curr_func = ca.interpolant(
            "v_curr", "linear", grid, array["water_v"].values.ravel(order="F")
        )

    # Plotting Functions for OceanCurrents specifically
    @staticmethod
    def plot_data_from_xarray(
            time_idx: int,
            xarray: xr,
            vmin: Optional[float] = 0,
            vmax: Optional[float] = None,
            alpha: Optional[float] = 0.5,
            plot_type: Optional[AnyStr] = "quiver",
            colorbar: Optional[bool] = True,
            ax: Optional[matplotlib.pyplot.axes] = None,
            fill_nan: Optional[bool] = True,
            return_cbar: Optional[bool] = False,
            set_title: Optional[bool] = True
    ) -> matplotlib.pyplot.axes:
        """Base function to plot the currents from an xarray. If xarray has a time-dimension time_idx is selected,
        if xarray's time dimension is already collapsed (e.g. after interpolation) it's directly plotted.
        All other functions build on top of it, it creates the ax object and returns it.
        Args:
            time_idx:          time-idx to select from the xarray (only if it has time dimension)
            xarray:            xarray object containing the grids and data
            plot_type:         a string specifying the plot type: streamline or quiver
            vmin:              minimum current magnitude used for colorbar (float)
            vmax:              maximum current magnitude used for colorbar (float)
            alpha:             alpha of the current magnitude color visualization
            colorbar:          if to plot the colorbar or not
            ax:                Optional for feeding in an axis object to plot the figure on.
            fill_nan:          If True NaN will be filled with 0, otherwise left as NaN
            return_cbar:       if True the colorbar object is returned
        Returns:
            ax                 matplotlib.pyplot.axes object
        """
        if fill_nan:
            xarray = xarray.fillna(0)
        # Step 1: Make the data ready for plotting
        # check if time-dimension already collapsed or not yet
        if xarray["time"].size != 1:
            xarray = xarray.isel(time=time_idx)
        # calculate magnitude if not in there yet
        if "magnitude" not in xarray.keys():
            xarray = xarray.assign(magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5)
        time = get_datetime_from_np64(xarray["time"].data)

        # Step 2: Create ax object
        if ax is None:
            ax = plt.axes()
        if set_title:
            ax.set_title("Time: " + time.strftime("%Y-%m-%d %H:%M:%S UTC"))

        # underly with current magnitude
        if vmax is None:
            vmax = np.max(xarray["magnitude"].max()).item()
        im = xarray["magnitude"].plot(
            cmap="jet", vmin=vmin, vmax=vmax, alpha=alpha, ax=ax, add_colorbar=False
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
                ["{:.{prec}f}".format(t, prec=precision) for t in cbar.get_ticks().tolist()]
            )
        # Plot on ax object
        if plot_type == "streamline":
            # Needed because the data needs to be perfectly equally spaced
            time_2D_array = format_to_equally_spaced_xy_grid(xarray).fillna(0)
            time_2D_array.plot.streamplot(
                x="lon", y="lat", u="water_u", v="water_v", color="black", ax=ax
            )
            ax.set_ylim([time_2D_array["lat"].data.min(), time_2D_array["lat"].data.max()])
            ax.set_xlim([time_2D_array["lon"].data.min(), time_2D_array["lon"].data.max()])
        elif plot_type == "quiver":
            xarray.plot.quiver(x="lon", y="lat", u="water_u", v="water_v", ax=ax, add_guide=False)

        if return_cbar:
            return ax, cbar
        return ax

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
                f'Point {point} is not inside x_dict {self.grid_dict["x_grid"][[0, -1]]}'
            )
        if not self.grid_dict["y_grid"].min() < point.lat.deg < self.grid_dict["y_grid"].max():
            raise ValueError(
                f'Point {point} is not inside y_grid {self.grid_dict["y_grid"][[0, -1]]}'
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
            bool:     True if on land and false otherwise
        """
        if not self.grid_dict["x_grid"].min() < point.lon.deg < self.grid_dict["x_grid"].max():
            raise ValueError(
                f'Point {point} is not inside x_dict {self.grid_dict["x_grid"][[0, -1]]}'
            )
        if not self.grid_dict["y_grid"].min() < point.lat.deg < self.grid_dict["y_grid"].max():
            raise ValueError(
                f'Point {point} is not inside y_grid {self.grid_dict["y_grid"][[0, -1]]}'
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
        # print('__del__ called in OceanCurrentSource')
        del self.u_curr_func
        del self.v_curr_func
        pass


class OceanCurrentSourceXarray(OceanCurrentSource, XarraySource):
    """Class for OceanCurrentSources that are based on Xarray in contrast to an analytical function.
    It is the base-class for the various hindcast and forecast data sources."""

    def __init__(self, source_config_dict: dict):
        """See init of OceanCurrentSource, XarraySource  for details."""
        super().__init__(source_config_dict)
        # initialize logger
        self.logger = logging.getLogger("arena.ocean_field.ocean_source")
        self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
        self.u_curr_func, self.v_curr_func = [None] * 2
        self.dask_array = None

    def get_data_over_area(
            self,
            x_interval: List[float],
            y_interval: List[float],
            t_interval: List[datetime.datetime],
            spatial_resolution: Optional[float] = None,
            temporal_resolution: Optional[float] = None,
    ) -> xr:
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
        subset = super().get_data_over_area(
            x_interval, y_interval, t_interval, spatial_resolution, temporal_resolution
        )

        # Step 2: make explicit
        subset = self.make_explicit(subset)

        return subset

    def make_explicit(self, dataframe: xr) -> xr:
        """Helper function to handle that multi-file access needs compute to be made explicit."""
        if self.dask_array:
            dataframe = dataframe.compute()
        return dataframe

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> OceanCurrentVector:
        data_xarray = self.make_explicit(super().get_data_at_point(spatio_temporal_point))
        return OceanCurrentVector(u=data_xarray["water_u"].item(), v=data_xarray["water_v"].item())


class ForecastFileSource(OceanCurrentSourceXarray):
    # TODO: Make it work with multiple files for one forecast (a bit of extra logic, but possible)
    """Data Source Object that accesses and manages multiple daily HYCOM files as source."""

    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        # Step 1: get the dictionary of all files from the specific folder
        self.files_dicts = get_file_dicts(
            source_config_dict["source_settings"]["folder"],
            currents=source_config_dict["source_settings"].get("currents", "normal"),
        )

        # Step 2: derive the time coverage and grid_dict for from the first file
        self.t_forecast_coverage = [
            self.files_dicts[0]["t_range"][0],  # t_0 of fist forecast file
            self.files_dicts[-1]["t_range"][1],
        ]  # t_final of last forecast file

        self.grid_dict = self.files_dicts[0]

        # stateful variable to prevent checking for most current FMRC file for every forecast
        self.rec_file_idx = 0
        self.load_ocean_current_from_idx()

    def get_data_over_area(
            self,
            x_interval: List[float],
            y_interval: List[float],
            t_interval: List[Union[datetime.datetime, int]],
            spatial_resolution: Optional[float] = None,
            temporal_resolution: Optional[float] = None,
            most_recent_fmrc_at_time: Optional[datetime.datetime] = None,
    ) -> xr:
        # format to datetime object
        if not isinstance(t_interval[0], datetime.datetime):
            t_interval = [
                datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)
                for time in t_interval
            ]
        # Step 1: Make sure we use the right forecast either most_recent_fmrc_at_time or default t_interval[0]
        self.check_for_most_recent_fmrc_dataframe(
            most_recent_fmrc_at_time if most_recent_fmrc_at_time is not None else t_interval[0]
        )

        # Step 2: Return Subset
        return super().get_data_over_area(
            x_interval,
            y_interval,
            t_interval,
            spatial_resolution=spatial_resolution,
            temporal_resolution=temporal_resolution,
        )

    def load_ocean_current_from_idx(self):
        """Helper Function to load an OceanCurrent object."""
        self.DataArray = format_xarray(
            data_frame=xr.open_dataset(self.files_dicts[self.rec_file_idx]["file"]),
            currents=self.source_config_dict["source_settings"].get("currents", "total"),
        )

    def check_for_most_recent_fmrc_dataframe(self, time: datetime.datetime) -> int:
        """Helper function to check update the self.OceanCurrent if a new forecast is available at
        the specified input time.
        Args:
          time: datetime object
        """
        # check if rec_file_idx is already the last one and time is larger than its start time
        if (
                self.rec_file_idx + 1 == len(self.files_dicts)
                and self.files_dicts[self.rec_file_idx]["t_range"][0] <= time
        ):
            if time > self.files_dicts[self.rec_file_idx]["t_range"][1]:
                raise ValueError("No current data in the last file for requested time.")
            else:
                return self.files_dicts[self.rec_file_idx]["t_range"][0]
        # otherwise check if a more recent one is available or we need to use an older one
        elif not (
                self.files_dicts[self.rec_file_idx]["t_range"][0]
                <= time
                < self.files_dicts[self.rec_file_idx + 1]["t_range"][0]
        ):
            # Filter on the list to get all files where t_0 is contained.
            dics_containing_t_0 = list(
                filter(lambda dic: dic["t_range"][0] < time < dic["t_range"][1], self.files_dicts)
            )
            # Basic Sanity Check if this list is empty no file contains t_0
            if len(dics_containing_t_0) == 0:
                raise ValueError(f"None of the forecast files contains t_0={time}.")
            # As the dict is time-ordered we simple need to find the idx of the last one in the dics_containing_t_0
            for idx, dic in enumerate(self.files_dicts):
                if dic["t_range"][0] == dics_containing_t_0[-1]["t_range"][0]:
                    self.rec_file_idx = idx
            # set the new self.OceanCurrent
            self.load_ocean_current_from_idx()

        # always return the most current idx
        return self.files_dicts[self.rec_file_idx]["t_range"][0]

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> OceanCurrentVector:
        """We overwrite it because we don't want that Forecast needs caching..."""
        # Step 1: Make sure we use the most recent forecast available
        self.check_for_most_recent_fmrc_dataframe(spatio_temporal_point.date_time)
        return super().get_data_at_point(spatio_temporal_point)


class HindcastFileSource(OceanCurrentSourceXarray):
    """Data Source Object that accesses and manages one or many HYCOM files as source."""

    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        # Step 1: get the dictionary of all files from the specific folder
        self.files_dicts = get_file_dicts(source_config_dict["source_settings"]["folder"])

        # Step 2: open the respective file as multi dataset
        self.DataArray = format_xarray(
            data_frame=xr.open_mfdataset([h_dict["file"] for h_dict in self.files_dicts]),
            currents=source_config_dict["source_settings"].get("currents", "normal"),
        )

        # Step 3: Check if multi-file (then dask) or not
        self.dask_array = isinstance(self.DataArray["water_u"].data, dask.array.core.Array)

        # Step 4: derive the grid_dict for the xarray
        self.grid_dict = self.get_grid_dict_from_xr(self.DataArray)

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> OceanCurrentVector:
        return OceanCurrentVector(
            u=self.u_curr_func(spatio_temporal_point.to_spatio_temporal_casadi_input()),
            v=self.v_curr_func(spatio_temporal_point.to_spatio_temporal_casadi_input()),
        )


class HindcastOpendapSource(OceanCurrentSourceXarray):
    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        # Step 1: establish the opendap connection with the settings in config_dict
        if source_config_dict["source_settings"]["service"] == "copernicus":
            self.DataArray = xr.open_dataset(
                copernicusmarine_datastore(
                    source_config_dict["source_settings"]["DATASET_ID"],
                    source_config_dict["source_settings"]["USERNAME"],
                    source_config_dict["source_settings"]["PASSWORD"],
                )
            )
            self.DataArray = format_xarray(
                self.DataArray,
                currents=source_config_dict["source_settings"].get("currents", "normal"),
            )
        else:
            raise ValueError("Only opendap Copernicus implemented for now, HYCOM also has opendap.")

        # Step 2: derive the grid_dict for the xarray
        self.grid_dict = self.get_grid_dict_from_xr(
            self.DataArray, self.source_config_dict["source"]
        )

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> OceanCurrentVector:
        return OceanCurrentVector(
            u=self.u_curr_func(spatio_temporal_point.to_spatio_temporal_casadi_input()),
            v=self.v_curr_func(spatio_temporal_point.to_spatio_temporal_casadi_input()),
        )


# Helper functions across the OceanCurrentSource objects
def get_file_dicts(folders: Union[AnyStr, List[str]], currents="normal") -> List[dict]:
    """Creates an list of dicts ordered according to time available, one for each nc file available in folder.
    The dicts for each file contains:
    {'t_range': [<datetime object>, T], 'file': <filepath> ,'y_range': [min_lat, max_lat], 'x_range': [min_lon, max_lon]}
    """
    # get a list of files from the folder
    if type(folders) == str:
        folders = [folders]
    files_list = [
        folder + f
        for folder in folders
        for f in os.listdir(folder)
        if (os.path.isfile(os.path.join(folder, f)) and f != ".DS_Store")
    ]

    # iterate over all files to extract the grids and put them in an ordered list of dicts
    list_of_dicts = []
    for file in files_list:
        grid_dict = get_grid_dict_from_file(file, currents=currents)
        # append the file to it:
        grid_dict["file"] = file
        list_of_dicts.append(grid_dict)
    # sort the list
    list_of_dicts.sort(key=lambda dict: dict["t_range"][0])
    return list_of_dicts


def format_xarray(data_frame: xr, currents: AnyStr = "total") -> xr:
    """Helper Function to format Data Arrays consistently.
    Args:
          data_frame: data_frame object
          currents: String either 'normal' then uo, vo from Copernicus is used or
                    'total' then the total including tidal and wave drift is used.
    """
    # select the 0 depth dimension
    if "depth" in data_frame.coords:
        data_frame = data_frame.isel(depth=0)

    # format and name variables and dimensions consistenly across sources
    if "source" not in data_frame.attrs or "HYCOM" in data_frame.attrs["source"]:
        data_frame["time"] = data_frame["time"].dt.round("H")
        return data_frame
    elif "MERCATOR" in data_frame.attrs["source"]:
        # for consistency we need to rename the variables in the xarray the same as in hycom
        data_frame = data_frame.rename({"latitude": "lat", "longitude": "lon"})
        if currents == "total":
            return data_frame[["utotal", "vtotal"]].rename(
                {"utotal": "water_u", "vtotal": "water_v"}
            )
        elif currents == "normal":
            return data_frame[["uo", "vo"]].rename({"uo": "water_u", "vo": "water_v"})
    # for the generated noise fields
    elif "u_error" in data_frame.keys():
        # Note: a hack because of Jonas file, need to change that!
        data_frame = data_frame.transpose("time", "lat", "lon")
        return data_frame[["u_error", "v_error"]].rename(
            {"u_error": "water_u", "v_error": "water_v"}
        )


def get_grid_dict_from_file(file: AnyStr, currents="total") -> dict:
    """Helper function to create a grid dict from a local nc3 file."""
    f = format_xarray(xr.open_dataset(file), currents=currents)
    # get the time coverage in POSIX
    t_grid = get_posix_time_from_np64(f.variables["time"].data)
    y_range = [f.variables["lat"].data[0], f.variables["lat"].data[-1]]
    x_range = [f.variables["lon"].data[0], f.variables["lon"].data[-1]]
    # close netCDF file
    f.close()
    # create dict
    return {
        "t_range": [
            datetime.datetime.fromtimestamp(t_grid[0], datetime.timezone.utc),
            datetime.datetime.fromtimestamp(t_grid[-1], datetime.timezone.utc),
        ],
        "y_range": y_range,
        "x_range": x_range,
    }


def copernicusmarine_datastore(dataset, username, password):
    """Helper Function to establish an opendap session with copernicus data."""
    cas_url = "https://cmems-cas.cls.fr/cas/login"
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()["CASTGC"])
    database = ["my", "nrt"]
    url = f"https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}"
    try:
        data_store = xr.backends.PydapDataStore(
            open_url(url, session=session)
        )  # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
    except BaseException:
        url = f"https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}"
        data_store = xr.backends.PydapDataStore(
            open_url(url, session=session)
        )  # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
    return data_store


def format_to_equally_spaced_xy_grid(xarray):
    """Helper Function to format an xarray to equally spaced lat, lon axis."""
    xarray["lon"] = np.linspace(
        xarray["lon"].data[0], xarray["lon"].data[-1], len(xarray["lon"].data)
    )
    xarray["lat"] = np.linspace(
        xarray["lat"].data[0], xarray["lat"].data[-1], len(xarray["lat"].data)
    )
    return xarray
