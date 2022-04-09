import abc
from datetime import datetime, timedelta, timezone
from typing import List, NamedTuple, Sequence, AnyStr, Optional
from ocean_navigation_simulator.env.utils.units import get_posix_time_from_np64, get_datetime_from_np64
from ocean_navigation_simulator.env.data_sources.DataField import DataField
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


class OceanCurrentSource:
    """Base class for various data sources of ocean currents to handle different current sources."""

    def __init__(self, source_config_dict: dict):
        """Function to get the OceanCurrentVector at a specific point.
        Args:
          source_config_dict: TODO: detail what needs to be specified here
          """
        self.source_config_dict = source_config_dict
        self.grid_dict, self.dask_array,  self.OceanCurrents = [None]*3

    def get_currents_at_point(self, point: List[float], time: datetime) -> OceanCurrentVector:
        """Function to get the OceanCurrentVector at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          OceanCurrentVector
          """
        # Step 1: run interpolation via xarray
        currents_at_point = self.OceanCurrents.interp(time=np.datetime64(time),
                                                      lon=point[0], lat=point[1], method='linear')
        # Step 2: make explicit
        currents_at_point = self.make_explicit(currents_at_point)

        u = currents_at_point['water_u'].data.item()
        v = currents_at_point['water_u'].data.item()

        if np.any(np.isnan([u,v])):
            raise ValueError("Ocean current values at {} are nan, likely requested out of bound of the dataset.".format(point))

        return OceanCurrentVector(u=u, v=v)

    def get_currents_over_area(self, x_interval: List[float], y_interval: List[float],
                               t_interval: List[datetime],
                               spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None) -> xr:
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
        # Step 1: Subset the xarray accordingly
        subset = self.OceanCurrents.sel(
            time=slice(t_interval[0].replace(tzinfo=None) - timedelta(seconds=self.source_config_dict['subset_time_buffer_in_s']),
                       t_interval[1].replace(tzinfo=None) + timedelta(seconds=self.source_config_dict['subset_time_buffer_in_s'])),
            lon=slice(x_interval[0], x_interval[1]),
            lat=slice(y_interval[0], y_interval[1]))

        # Step 2: make explicit
        subset = self.make_explicit(subset)

        # Step 3: Do a sanity check for the sub-setting before it's used outside and leads to errors
        DataField.array_subsetting_sanity_check(subset, x_interval, y_interval, t_interval)

        # Step 4: perform interpolation to a specific resolution if requested
        if spatial_resolution is not None or temporal_resolution is not None:
            subset = DataField.interpolate_in_space_and_time(subset, spatial_resolution, temporal_resolution)

        return subset

    def make_explicit(self, dataframe: xr) -> xr:
        """Helper function to handle that multi-file access needs compute to be made explicit."""
        if self.dask_array:
            dataframe = dataframe.compute()
        return dataframe


class ForecastFileSource(OceanCurrentSource):
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

        self.grid_dict = derive_grid_dict_from_file_dict(self.files_dicts[0])

        # stateful variable to prevent checking for most current FMRC file for every forecast
        self.rec_file_idx = 0
        self.load_ocean_current_from_idx()

    def get_currents_at_point(self, point: List[float], time: datetime) -> OceanCurrentVector:
        # Step 1: Make sure we use the most recent forecast available
        self.check_for_most_recent_fmrc_dataframe(time)

        return super().get_currents_at_point(point, time)

    def get_currents_over_area(self, x_interval: List[float], y_interval: List[float],
                               t_interval: List[datetime],
                               spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None) -> xr:
        # Step 1: Make sure we use the most recent forecast available
        self.check_for_most_recent_fmrc_dataframe(t_interval[0])

        return super().get_currents_over_area(x_interval, y_interval, t_interval)

    def load_ocean_current_from_idx(self):
        """Helper Function to load an OceanCurrent object."""
        self.OceanCurrents = open_formatted_xarray(self.files_dicts[self.rec_file_idx]['file'])

    def check_for_most_recent_fmrc_dataframe(self, time: datetime) -> None:
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


class HindcastFileSource(OceanCurrentSource):
    """Data Source Object that accesses and manages one or many HYCOM files as source."""
    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        # Step 1: get the dictionary of all files from the specific folder
        self.files_dicts = get_file_dicts(source_config_dict['source_settings']['folder'])

        # Step 2: open the respective file as multi dataset
        self.OceanCurrents = xr.open_mfdataset([h_dict['file'] for h_dict in self.files_dicts]).isel(depth=0)
        self.OceanCurrents["time"] = self.OceanCurrents["time"].dt.round("H")

        # Step 3: Check if multi-file (then dask) or not
        self.dask_array = isinstance(self.OceanCurrents['water_u'].data, dask.array.core.Array)

        # Step 4: derive the grid_dict for the xarray
        self.grid_dict = get_grid_dict_from_xr(self.OceanCurrents)


class HindcastOpendapSource(OceanCurrentSource):
    def __init__(self,  source_config_dict: dict):
        super().__init__(source_config_dict)
        # Step 1: establish the opendap connection with the settings in config_dict
        if source_config_dict['source_settings']['service'] == 'copernicus':
            self.OceanCurrents = xr.open_dataset(copernicusmarine_datastore(source_config_dict['source_settings']['DATASET_ID'],
                                                                            source_config_dict['source_settings']['USERNAME'],
                                                                            source_config_dict['source_settings']['PASSWORD'])).isel(depth=0)
            # for consistency we need to rename the variables in the xarray the same as in hycom
            self.OceanCurrents = self.OceanCurrents.rename({'latitude': 'lat', 'longitude': 'lon'})
            if source_config_dict['source_settings']['currents'] == 'total':
                self.OceanCurrents = self.OceanCurrents[['utotal', 'vtotal']].rename({'utotal': 'water_u', 'vtotal': 'water_v'})
            elif source_config_dict['source_settings']['currents'] == 'normal':
                self.OceanCurrents = self.OceanCurrents[['uo', 'vo']].rename({'uo': 'water_u', 'vo': 'water_v'})
        else:
            raise ValueError("Only opendap Copernicus implemented for now.")

        # Step 2: derive the grid_dict for the xarray
        self.grid_dict = get_grid_dict_from_xr(self.OceanCurrents)





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


def derive_grid_dict_from_file_dict(file_dict: dict) -> dict:
    """Helper function to create the a grid dict from one nc file_dict."""

    # get spatial coverage by reading in the first file
    f = open_formatted_xarray(file_dict['file'])
    xgrid = f.variables['lon'].data
    ygrid = f.variables['lat'].data

    # get the land mask
    u_data = f.variables['water_u'].data
    # adds a mask where there's a nan (on-land)
    u_data = np.ma.masked_invalid(u_data)

    return {"t_range": file_dict['t_range'], "y_range": file_dict['y_range'], "x_range": file_dict['x_range'],
            'spatial_land_mask': u_data[0, :, :].mask, 'x_grid': xgrid, 'y_grid': ygrid}


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
    return {"t_range": [datetime.fromtimestamp(t_grid[0], timezone.utc),
                        datetime.fromtimestamp(t_grid[-1], timezone.utc)],
            "y_range": y_range,
            "x_range": x_range}


def get_grid_dict_from_xr(xrDF: xr) -> dict:
    """Helper function to extract the grid dict from an xrarray"""

    grid_dict = {"t_range": [get_datetime_from_np64(np64) for np64 in [xrDF["time"].data[0], xrDF["time"].data[-1]]],
                 "y_range": [xrDF["lat"].data[0], xrDF["lat"].data[-1]],
                  'y_grid': xrDF["lat"].data,
                  "x_range": [xrDF["lon"].data[0], xrDF["lon"].data[-1]],
                  'x_grid': xrDF["lon"].data,
                  # 'spatial_land_mask': np.ma.masked_invalid(xrDF.variables['water_u'].data[0, :, :]).mask
                }

    return grid_dict


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
