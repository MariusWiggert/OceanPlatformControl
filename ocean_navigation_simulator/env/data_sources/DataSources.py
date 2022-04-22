"""The abstract base class for all Data Sources.
Implements a lot of shared functionality such as
"""

from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatioTemporalPoint, SpatialPoint
import warnings
import datetime
from typing import List, NamedTuple, Sequence, AnyStr, Optional, Tuple, Union
import numpy as np
import xarray as xr
import abc


class DataSource(abc.ABC):
    """Base class for various data sources."""

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
            """
        out_x_range = not (self.casadi_grid_dict['x_range'][0] < state.lon.deg < self.casadi_grid_dict['x_range'][1])
        out_y_range = not (self.casadi_grid_dict['y_range'][0] < state.lat.deg < self.casadi_grid_dict['y_range'][1])
        out_t_range = not (self.casadi_grid_dict['t_range'][0] <= state.date_time < self.casadi_grid_dict['t_range'][1])

        if out_x_range or out_y_range or out_t_range:
            if out_x_range:
                print(f'Updating Interpolation (X: {self.casadi_grid_dict["x_range"][0]}, {state.lon.deg}, {self.casadi_grid_dict["x_range"][1]}')
            if out_y_range:
                print(f'Updating Interpolation (Y: {self.casadi_grid_dict["y_range"][0]}, {state.lat.deg}, {self.casadi_grid_dict["y_range"][1]}')
            if out_t_range:
                print(f'Updating Interpolation (T: {self.casadi_grid_dict["t_range"][0]}, {state.date_time}, {self.casadi_grid_dict["t_range"][1]}')

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
            x_0=state.to_spatio_temporal_point(), x_T=state.to_spatial_point(),
            deg_around_x0_xT_box=self.source_config_dict['casadi_cache_settings']['deg_around_x_t'],
            temp_horizon_in_s=self.source_config_dict['casadi_cache_settings']['time_around_x_t'])

        # Step 2: Get the data from itself and update casadi_grid_dict
        xarray = self.get_data_over_area(x_interval, y_interval, t_interval)
        self.casadi_grid_dict = self.get_grid_dict_from_xr(xarray)

        # Step 3: Set up the grid
        grid = [
            units.get_posix_time_from_np64(xarray.coords['time'].values),
            xarray.coords['lat'].values,
            xarray.coords['lon'].values
        ]

        self.initialize_casadi_functions(grid, xarray)

    @staticmethod
    def convert_to_x_y_time_bounds(x_0: SpatioTemporalPoint, x_T: SpatialPoint,
                                   deg_around_x0_xT_box: float, temp_horizon_in_s: float):
        """ Helper function for spatio-temporal subsetting
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
        lon_bnds = [min(x_0.lon.deg, x_T.lon.deg) - deg_around_x0_xT_box, max(x_0.lon.deg, x_T.lon.deg) + deg_around_x0_xT_box]
        lat_bnds = [min(x_0.lat.deg, x_T.lat.deg) - deg_around_x0_xT_box, max(x_0.lat.deg, x_T.lat.deg) + deg_around_x0_xT_box]

        return t_interval, lat_bnds, lon_bnds

    @staticmethod
    def get_grid_dict_from_xr(xrDF: xr) -> dict:
        """Helper function to extract the grid dict from an xrarray"""

        grid_dict = {
            "t_range": [units.get_datetime_from_np64(np64) for np64 in [xrDF["time"].data[0], xrDF["time"].data[-1]]],
            "y_range": [xrDF["lat"].data[0], xrDF["lat"].data[-1]],
            'y_grid': xrDF["lat"].data,
            "x_range": [xrDF["lon"].data[0], xrDF["lon"].data[-1]],
            'x_grid': xrDF["lon"].data,
            't_grid': [units.get_posix_time_from_np64(np64) for np64 in xrDF["time"].data]
            # 'spatial_land_mask': np.ma.masked_invalid(xrDF.variables['water_u'].data[0, :, :]).mask
            }

        return grid_dict

    @staticmethod
    def array_subsetting_sanity_check(array: xr, x_interval: List[float], y_interval: List[float],
                                      t_interval: List[datetime.datetime]):
        """Advanced Check if admissible subset and warning of partially being out of bound in space or time."""
        # Step 1: collateral check is any dimension 0?
        if 0 in (len(array.coords['lat']), len(array.coords['lon']), len(array.coords['time'])):
            # check which dimension for more informative errors
            if len(array.coords['time']) == 0:
                raise ValueError("None of the requested t_interval is in the file.")
            else:
                raise ValueError("None of the requested spatial area is in the file.")
        if units.get_datetime_from_np64(array.coords['time'].data[0]) > t_interval[0]:
            raise ValueError("The starting time {} is not in the array.".format(t_interval[0]))

        # Step 2: Data partially not in the array check
        if array.coords['lat'].data[0] > y_interval[0] or array.coords['lat'].data[-1] < y_interval[1]:
            warnings.warn("Part of the y requested area is outside of file.", RuntimeWarning)
        if array.coords['lon'].data[0] > x_interval[0] or array.coords['lon'].data[-1] < x_interval[1]:
            warnings.warn("Part of the x requested area is outside of file.", RuntimeWarning)
        if units.get_datetime_from_np64(array.coords['time'].data[-1]) < t_interval[1]:
            warnings.warn("The final time is not part of the subset.".format(t_interval[1]), RuntimeWarning)


# Two types of data sources: analytical and xarray based ones -> need different default functions, used via mixin
class XarraySource(abc.ABC):
    def __init__(self, source_config_dict: dict):
        self.source_config_dict = source_config_dict
        self.DataArray = None  # The xarray containing the raw data (if not an analytical function)
        self.grid_dict, self.casadi_grid_dict = [None] * 2

    def get_data_at_point(self, point: List[float], time: datetime.datetime) -> xr:
        """Function to get the data at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          xr object that is then processed by the respective data source for its purpose
          """
        return self.DataArray.interp(time=np.datetime64(time), lon=point[0], lat=point[1], method='linear')

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
        # Step 1: Subset the xarray accordingly
        # Step 1.1 include buffers around of the grid granularity (assuming regular grid axis)
        dt = self.DataArray['time'][1] - self.DataArray['time'][0]
        t_interval_extended = [np.datetime64(t_interval[0].replace(tzinfo=None)) - dt,
                               np.datetime64(t_interval[1].replace(tzinfo=None)) + dt]
        dlon = self.DataArray['lon'][1] - self.DataArray['lon'][0]
        x_interval_extended = [x_interval[0] - dlon, x_interval[1] + dlon]
        dlat = self.DataArray['lat'][1] - self.DataArray['lat'][0]
        y_interval_extended = [y_interval[0] - dlat, y_interval[1] + dlat]
        subset = self.DataArray.sel(
            time=slice(t_interval_extended[0], t_interval_extended[1]),
            lon=slice(x_interval_extended[0], x_interval_extended[1]),
            lat=slice(y_interval_extended[0], y_interval_extended[1]))

        # Step 3: Do a sanity check for the sub-setting before it's used outside and leads to errors
        DataSource.array_subsetting_sanity_check(subset, x_interval, y_interval, t_interval)

        # Step 4: perform interpolation to a specific resolution if requested
        if spatial_resolution is not None or temporal_resolution is not None:
            subset = self.interpolate_in_space_and_time(subset, spatial_resolution, temporal_resolution)

        return subset

    @staticmethod
    def interpolate_in_space_and_time(array: xr, spatial_resolution: Optional[float],
                                      temporal_resolution: Optional[float]) -> xr:
        """Helper function for temporal and spatial interpolation"""
        # Run temporal interpolation
        if temporal_resolution is not None:
            time_grid = np.arange(start=array['time'][0].data, stop=array['time'][-1].data,
                                  step=np.timedelta64(temporal_resolution, 's'))
            array = array.interp(time=time_grid, method='linear')

        # Run spatial interpolation
        if spatial_resolution is not None:
            lat_grid = np.arange(start=array['lat'][0].data, stop=array['lat'][-1].data,
                                 step=spatial_resolution)
            lon_grid = np.arange(start=array['lon'][0].data, stop=array['lon'][-1].data,
                                 step=spatial_resolution)
            array = array.interp(
                lon=lon_grid,
                lat=lat_grid, method='linear')

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
        if 'boundary_buffers' not in source_config_dict['source_settings']:
            self.spatial_boundary_buffers = np.array([0., 0.])
        else:
            self.spatial_boundary_buffers = np.array(source_config_dict['source_settings']['boundary_buffers'])
        self.x_domain = [x + self.spatial_boundary_buffers[0] * (-1) ** i for x, i in zip(source_config_dict['source_settings']['x_domain'], [1, 2])]
        self.y_domain = [y + self.spatial_boundary_buffers[1] * (-1) ** i for y, i in zip(source_config_dict['source_settings']['y_domain'], [1, 2])]
        # set the temp_domain_posix
        if isinstance(source_config_dict['source_settings']['temporal_domain'][0], datetime.datetime):
            # Assume it's utc if not correct it
            self.temp_domain_posix = [t.replace(tzinfo=datetime.timezone.utc).timestamp()
                                 for t in source_config_dict['source_settings']['temporal_domain']]
        else:
            self.temp_domain_posix = source_config_dict['source_settings']['temporal_domain']
        # Set the default resolutions (used when none is provided in get_data_over_area)
        self.spatial_resolution = source_config_dict['source_settings']['spatial_resolution']
        self.temporal_resolution = source_config_dict['source_settings']['temporal_resolution']

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
    def get_data_at_point(self, point: List[float], time: datetime) -> xr:
        """Function to get the data at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          xr object that is then processed by the respective data source for its purpose
          """
        raise NotImplementedError

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

        # Step 0.0: if t_interval is in datetime convert to POSIX
        if isinstance(t_interval[0], datetime.datetime):
            t_interval_posix = [time.timestamp() for time in t_interval]
        else:
            t_interval_posix = t_interval

        # Get the coordinate vectors to calculate the analytical function over
        grids_dict = self.get_grid_dict(x_interval, y_interval, t_interval_posix,
                                        spatial_resolution=spatial_resolution, temporal_resolution=temporal_resolution)

        data_tuple = self.map_analytical_function_over_area(grids_dict)

        # make an xarray object out of it
        subset = self.create_xarray(grids_dict, data_tuple)

        # Step 3: Do a sanity check for the sub-setting before it's used outside and leads to errors
        DataSource.array_subsetting_sanity_check(subset, x_interval, y_interval, t_interval)

        return subset

    def get_ranges_dict(self, x_interval: Optional[List[float]] = None, y_interval: Optional[List[float]] = None,
                        t_interval: Optional[List[float]] = None):
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
            t_interval = [max(t_interval[0], self.temp_domain_posix[0]),
                          min(t_interval[1], self.temp_domain_posix[1])]
        if x_interval is None:
            x_interval = [self.x_domain[0], self.x_domain[1]]
        else:
            x_interval = [max(x_interval[0], self.x_domain[0]),
                          min(x_interval[1], self.x_domain[1])]
        if y_interval is None:
            y_interval = [self.y_domain[0], self.y_domain[1]]
        else:
            y_interval = [max(y_interval[0], self.y_domain[0]),
                          min(y_interval[1], self.y_domain[1])]

        return {"y_range": y_interval, "x_range": x_interval,
                "t_range": [datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc) for t in t_interval]}

    def get_grid_dict(self, x_interval: Optional[List[float]] = None, y_interval: Optional[List[float]] = None,
                      t_interval: Optional[List[float]] = None,
                      spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None):

        """Helper Function to produce a grid dict."""
        if spatial_resolution is None:
            spatial_resolution = self.spatial_resolution
        if temporal_resolution is None:
            temporal_resolution = self.temporal_resolution

        # Step 1: Get the domain adjusted range dict
        ranges_dict = self.get_ranges_dict(x_interval, y_interval, t_interval)

        # Step 2: Get the grids with the respective resolutions
        # Step 2.1 Spatial coordinate vectors with desired resolution
        lo_hi_vec = [[lon, lat] for lon, lat in zip(ranges_dict['x_range'], ranges_dict['y_range'])]
        # The + spatial_resolution is a hacky way to include the endpoint. We want a regular grid hence the floor to two decimals.
        spatial_vectors = [np.arange(start=np.floor(l*100)/100, stop=h + spatial_resolution, step=spatial_resolution) for l, h in
                              zip(lo_hi_vec[0], lo_hi_vec[1])]
        # Step 2.2 Temporal grid in POSIX TIME. We want a regular grid hence the floor to two decimals.
        t_grid = np.arange(start=np.floor((t_interval[0] - temporal_resolution)*100)/100, stop=t_interval[1] + temporal_resolution, step=temporal_resolution)

        return {'x_grid': spatial_vectors[0],
                'y_grid': spatial_vectors[1],
                't_grid': t_grid}

    def is_boundary(self, lon: Union[float, np.array], lat: Union[float, np.array],
                    posix_time: Union[float, np.array]) -> Union[float, np.array]:
        """Helper function to check if a state is in the boundary specified in hj as obstacle."""
        x_boundary = np.logical_or(lon < self.x_domain[0] + self.spatial_boundary_buffers[0],
                                    lon > self.x_domain[1] - self.spatial_boundary_buffers[0])
        y_boundary = np.logical_or(lat < self.y_domain[0] + self.spatial_boundary_buffers[1],
                                    lat > self.y_domain[1] - self.spatial_boundary_buffers[1])

        return np.logical_or(x_boundary, y_boundary)
