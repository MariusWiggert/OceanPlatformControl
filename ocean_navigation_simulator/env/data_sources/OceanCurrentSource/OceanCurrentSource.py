import abc
import datetime
from typing import List, NamedTuple, Sequence, AnyStr, Optional, Union, Tuple
from ocean_navigation_simulator.env.utils.units import get_posix_time_from_np64, get_datetime_from_np64
from ocean_navigation_simulator.env.data_sources.DataField import DataField
import casadi as ca
import jax
import matplotlib.pyplot as plt
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
from ocean_navigation_simulator.env.data_sources.DataSources import DataSource, XarraySource
from ocean_navigation_simulator.env.PlatformState import SpatioTemporalPoint


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
# TODO: Does not work well yet for getting the most recent forecast point data!


class OceanCurrentSource(DataSource):
    """Base class for various data sources of ocean currents to handle different current sources."""

    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        # Note: the input to the casadi function needs to be an array of the form np.array([posix time, lat, lon])
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """

        self.u_curr_func = ca.interpolant('u_curr', 'linear', grid, array['water_u'].values.ravel(order='F'))
        self.v_curr_func = ca.interpolant('v_curr', 'linear', grid, array['water_v'].values.ravel(order='F'))

    def plot_currents_at_time(self, time: Union[datetime.datetime, float], x_interval: List[float], y_interval: List[float],
                              plot_type: AnyStr = 'quiver', return_ax: Optional[bool] = False,
                              vmin: Optional[float] = 0, vmax: Optional[float] = None,
                              alpha: Optional[float] = 0.5, target_max_n: Optional[int] = None,
                              reset_plot: Optional[bool] = False, figsize: Tuple[int] = (6, 6)):
        """Plot all plot_streamlines_at_time over a specific area.
        Args:
          time: timefor which to plot the data either posix or datetime.datetime object
          x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          plot_type:       a string specifying the plot type: streamline or quiver
          return_ax: if True returns ax, otherwise renders plots with plt.show()
        """
        # reset plot this is needed for matplotlib.animation
        if reset_plot:
            plt.clf()
        else:  # create a new figure object where this is plotted
            fig = plt.figure(figsize=figsize)

        # format to datetime object
        if not isinstance(time, datetime.datetime):
            time = datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)

        # Step 0: check if we need to adjust spatial_resolution
        spatial_res = None
        if target_max_n is not None:
            n_x = (x_interval[1] - x_interval[0]) / self.grid_dict['spatial_res']
            n_y = (y_interval[1] - y_interval[0]) / self.grid_dict['spatial_res']
            max_n = max(n_y, n_x)
            # adjust temporal resolution
            if max_n > target_max_n:
                spatial_res = (max_n / target_max_n) * self.grid_dict['spatial_res']

        # Step 1: get the area data
        area_xarray = self.get_data_over_area(x_interval=x_interval,
                                              y_interval=y_interval,
                                              t_interval=[time, time + datetime.timedelta(seconds=1)],
                                              spatial_resolution=spatial_res)
        # Interpolate to the specific point
        time_2D_array = area_xarray.interp(time=time.replace(tzinfo=None))
        # calculate magnitude
        time_2D_array = time_2D_array.assign(magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2)** 0.5)

        # Step 2: Create ax object
        if self.source_config_dict['use_geographic_coordinate_system'] and plot_type == 'quiver':
            ax = self.set_up_geographic_ax()
            ax.set_title("Time: " + time.strftime('%Y-%m-%d %H:%M:%S UTC'))
        else:  # Non-dimensional
            ax = plt.axes()
            ax.set_title("Time: {time:.2f}".format(time=time.timestamp()))

        # underly with current magnitude
        if vmax is None:
            vmax = np.max(time_2D_array['magnitude'].max())
        time_2D_array['magnitude'].plot(cmap='jet', vmin=vmin, vmax=vmax, alpha=alpha, ax=ax)
        # set and format colorbar
        cbar = ax.collections[-1].colorbar
        cbar.ax.set_ylabel('current velocity')
        cbar.set_ticks(cbar.get_ticks())
        cbar.set_ticklabels(["{:.1f}".format(l) + ' m/s' for l in cbar.get_ticks().tolist()])

        # Plot on ax object
        if plot_type == 'streamline':
            # Needed because the data needs to be perfectly equally spaced
            time_2D_array = format_to_equally_spaced_xy_grid(time_2D_array).fillna(0)
            time_2D_array.plot.streamplot(x='lon', y='lat', u='water_u', v='water_v', color='black', ax=ax)
            ax.set_ylim([time_2D_array['lat'].data.min(), time_2D_array['lat'].data.max()])
            ax.set_xlim([time_2D_array['lon'].data.min(), time_2D_array['lon'].data.max()])
        elif plot_type == 'quiver':
            time_2D_array.plot.quiver(x='lon', y='lat', u='water_u', v='water_v', ax=ax)

        # Label the title
        if self.source_config_dict['use_geographic_coordinate_system'] and plot_type == 'quiver':
            ax.set_title("Time: " + time.strftime('%Y-%m-%d %H:%M:%S UTC'))
        else:  # Non-dimensional
            ax.set_title("Time: {time:.2f}".format(time=time.timestamp()))

        if return_ax:
            return ax
        else:
            plt.show()


class OceanCurrentSourceXarray(OceanCurrentSource, XarraySource):
    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        self.u_curr_func, self.v_curr_func = [None] * 2
        self.dask_array = None

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

        # Step 2: make explicit
        subset = self.make_explicit(subset)

        return subset

    def make_explicit(self, dataframe: xr) -> xr:
        """Helper function to handle that multi-file access needs compute to be made explicit."""
        if self.dask_array:
            dataframe = dataframe.compute()
        return dataframe

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> OceanCurrentVector:
        """Function to get the OceanCurrentVector at a specific point using the interpolation functions.
        Args:
          spatio_temporal_point: SpatioTemporalPoint in the respective used coordinate system geospherical or unitless
        Returns:
          OceanCurrentVector
          """
        return OceanCurrentVector(u=self.u_curr_func(spatio_temporal_point.to_spatio_temporal_casadi_input()),
                                  v=self.v_curr_func(spatio_temporal_point.to_spatio_temporal_casadi_input()))


class ForecastFileSource(OceanCurrentSourceXarray):
    # TODO: Make it work with multiple Global HYCOM FMRC Files (a bit of extra logic, but possible)
    """Data Source Object that accesses and manages multiple daily HYCOM files as source."""

    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        # Step 1: get the dictionary of all files from the specific folder
        self.files_dicts = get_file_dicts(source_config_dict['source_settings']['folder'])

        # Step 2: derive the time coverage and grid_dict for from the first file
        self.t_forecast_coverage = [
            self.files_dicts[0]['t_range'][0],  # t_0 of fist forecast file
            self.files_dicts[-1]['t_range'][1]]  # t_final of last forecast file

        self.grid_dict = self.files_dicts[0]

        # stateful variable to prevent checking for most current FMRC file for every forecast
        self.rec_file_idx = 0
        self.load_ocean_current_from_idx()

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> OceanCurrentVector:
        # Step 1: Make sure we use the most recent forecast available
        self.check_for_most_recent_fmrc_dataframe(SpatioTemporalPoint.date_time)

        return super().get_data_at_point(spatio_temporal_point)

    def get_data_over_area(self, x_interval: List[float], y_interval: List[float],
                           t_interval: List[datetime.datetime],
                           spatial_resolution: Optional[float] = None,
                           temporal_resolution: Optional[float] = None) -> xr:
        # Step 1: Make sure we use the most recent forecast available
        self.check_for_most_recent_fmrc_dataframe(t_interval[0])

        return super().get_data_over_area(x_interval, y_interval, t_interval)

    def load_ocean_current_from_idx(self):
        """Helper Function to load an OceanCurrent object."""
        self.DataArray = open_formatted_xarray(self.files_dicts[self.rec_file_idx]['file'])

    def check_for_most_recent_fmrc_dataframe(self, time: datetime.datetime) -> None:
        """Helper function to check update the self.OceanCurrent if a new forecast is available at
        the specified input time.
        Args:
          time: datetime object
        """
        # check if rec_file_idx is already the last one and time is larger than its start time
        if self.rec_file_idx + 1 == len(self.files_dicts) and self.files_dicts[self.rec_file_idx]['t_range'][0] <= time:
            return None

        # otherwise check if a more recent one is available or we need to use an older one
        elif not (self.files_dicts[self.rec_file_idx]['t_range'][0] <=
                  time < self.files_dicts[self.rec_file_idx + 1]['t_range'][0]):
            # Filter on the list to get all files where t_0 is contained.
            dics_containing_t_0 = list(
                filter(lambda dic: dic['t_range'][0] < time < dic['t_range'][1], self.files_dicts))
            # Basic Sanity Check if this list is empty no file contains t_0
            if len(dics_containing_t_0) == 0:
                raise ValueError("None of the forecast files contains time.")
            # As the dict is time-ordered we simple need to find the idx of the last one in the dics_containing_t_0
            for idx, dic in enumerate(self.files_dicts):
                if dic['t_range'][0] == dics_containing_t_0[-1]['t_range'][0]:
                    self.rec_file_idx = idx
            # set the new self.OceanCurrent
            self.load_ocean_current_from_idx()


class HindcastFileSource(OceanCurrentSourceXarray):
    """Data Source Object that accesses and manages one or many HYCOM files as source."""

    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        # Step 1: get the dictionary of all files from the specific folder
        self.files_dicts = get_file_dicts(source_config_dict['source_settings']['folder'])

        # Step 2: open the respective file as multi dataset
        self.DataArray = xr.open_mfdataset([h_dict['file'] for h_dict in self.files_dicts]).isel(depth=0)
        self.DataArray["time"] = self.DataArray["time"].dt.round("H")

        # Step 3: Check if multi-file (then dask) or not
        self.dask_array = isinstance(self.DataArray['water_u'].data, dask.array.core.Array)

        # Step 4: derive the grid_dict for the xarray
        self.grid_dict = self.get_grid_dict_from_xr(self.DataArray)


class HindcastOpendapSource(OceanCurrentSourceXarray):
    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        # Step 1: establish the opendap connection with the settings in config_dict
        if source_config_dict['source_settings']['service'] == 'copernicus':
            self.DataArray = xr.open_dataset(
                copernicusmarine_datastore(source_config_dict['source_settings']['DATASET_ID'],
                                           source_config_dict['source_settings']['USERNAME'],
                                           source_config_dict['source_settings']['PASSWORD'])).isel(depth=0)
            # for consistency we need to rename the variables in the xarray the same as in hycom
            self.DataArray = self.DataArray.rename({'latitude': 'lat', 'longitude': 'lon'})
            if source_config_dict['source_settings']['currents'] == 'total':
                self.DataArray = self.DataArray[['utotal', 'vtotal']].rename(
                    {'utotal': 'water_u', 'vtotal': 'water_v'})
            elif source_config_dict['source_settings']['currents'] == 'normal':
                self.DataArray = self.DataArray[['uo', 'vo']].rename({'uo': 'water_u', 'vo': 'water_v'})
        else:
            raise ValueError("Only opendap Copernicus implemented for now.")

        # Step 2: derive the grid_dict for the xarray
        self.grid_dict = self.get_grid_dict_from_xr(self.DataArray)


# Helper functions across the OceanCurrentSource objects
def get_file_dicts(folder: AnyStr) -> List[dict]:
    """ Creates an list of dicts ordered according to time available, one for each nc file available in folder.
    The dicts for each file contains:
    {'t_range': [<datetime object>, T], 'file': <filepath> ,'y_range': [min_lat, max_lat], 'x_range': [min_lon, max_lon]}
    """
    # get a list of files from the folder
    files_list = [folder + f for f in os.listdir(folder) if
                  (os.path.isfile(os.path.join(folder, f)) and f != '.DS_Store')]

    # iterate over all files to extract the grids and put them in an ordered list of dicts
    list_of_dicts = []
    for file in files_list:
        grid_dict = get_grid_dict_from_file(file)
        # append the file to it:
        grid_dict['file'] = file
        list_of_dicts.append(grid_dict)
    # sort the list
    list_of_dicts.sort(key=lambda dict: dict['t_range'][0])
    return list_of_dicts


def open_formatted_xarray(filepath: AnyStr) -> xr:
    data_frame = xr.open_dataset(filepath).isel(depth=0)
    if 'HYCOM' in data_frame.attrs['source']:
        return data_frame
    elif 'MERCATOR' in data_frame.attrs['source']:
        # for consistency we need to rename the variables in the xarray the same as in hycom
        data_frame_renamed = data_frame.rename({'vtotal': 'water_v', 'utotal': 'water_u',
                                                'latitude': 'lat', 'longitude': 'lon'})
        return data_frame_renamed


def get_grid_dict_from_file(file: AnyStr) -> dict:
    """Helper function to create a grid dict from a local nc3 file."""
    f = open_formatted_xarray(file)
    # get the time coverage in POSIX
    t_grid = get_posix_time_from_np64(f.variables['time'].data)
    y_range = [f.variables['lat'].data[0], f.variables['lat'].data[-1]]
    x_range = [f.variables['lon'].data[0], f.variables['lon'].data[-1]]
    # close netCDF file
    f.close()
    # create dict
    return {"t_range": [datetime.datetime.fromtimestamp(t_grid[0], datetime.timezone.utc),
                        datetime.datetime.fromtimestamp(t_grid[-1], datetime.timezone.utc)],
            "y_range": y_range,
            "x_range": x_range}


def copernicusmarine_datastore(dataset, username, password):
    """Helper Function to establish an opendap session with copernicus data."""
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url,
                                                         session=session))  # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url,
                                                         session=session))  # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
    return data_store


def format_to_equally_spaced_xy_grid(xarray):
    """Helper Function to format an xarray to equally spaced lat, lon axis."""
    xarray['lon'] = np.linspace(xarray['lon'].data[0], xarray['lon'].data[-1],
                                       len(xarray['lon'].data))
    xarray['lat'] = np.linspace(xarray['lat'].data[0], xarray['lat'].data[-1],
                                       len(xarray['lat'].data))
    return xarray
