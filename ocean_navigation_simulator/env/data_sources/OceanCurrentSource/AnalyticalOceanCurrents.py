import datetime
import matplotlib.pyplot as plt
from typing import List, NamedTuple, Sequence, AnyStr, Optional, Tuple, Union
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


# TODO: How does the direction connection of the analytical function into HJ Planner work? Is it even desirable?
# -> it's possible but I would have to give up passing OceanCurrentVectors around but rather the numbers directly.


class OceanCurrentSourceAnalytical(OceanCurrentSource, AnalyticalSource):
    def __init__(self, source_config_dict: dict):
        super().__init__(source_config_dict)
        # Casadi functions are created and maintained here but used in the platform object
        self.u_curr_func, self.v_curr_func = [None] * 2
        self.current_run_t_0 = 0

    @abc.abstractmethod
    def u_current_analytical(self, lon: Union[float, np.array], lat: Union[float, np.array],
                             posix_time: Union[float, np.array]) -> Union[float, np.array]:
        """Calculating the current in longitudinal direction at a specific lat, lon point at posix_time.
        Note: this can be used for floats as input or np.arrays which are all the same shape.
        Args:
            lon: longitude in degree
            lat: latitude in degree
            posix_time: POSIX time
        Returns:
            u_currents     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        raise NotImplementedError

    @abc.abstractmethod
    def v_current_analytical(self, lon: Union[float, np.array], lat: Union[float, np.array],
                             posix_time: Union[float, np.array]) -> Union[float, np.array]:
        """Calculating the current in latitudinal direction at a specific lat, lon point at posix_time.
        Note: this can be used for floats as input or np.arrays which are all the same shape.
        Args:
            lon: longitude in degree
            lat: latitude in degree
            posix_time: POSIX time
        Returns:
            v_currents     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        raise NotImplementedError

    def create_xarray(self, grids_dict: dict, data_tuple: Tuple) -> xr:
        """Function to create an xarray from the data tuple and grid dict
            Args:
              data_tuple: tuple containing (data_u, data_v)
              grids_dict: containing ranges and grids of x, y, t dimension
            Returns:
              xr     an xarray containing both the grid and data
            """
        # make a xarray object out of it
        return xr.Dataset(
            dict(water_u=(["time", "lat", "lon"], data_tuple[0]), water_v=(["time", "lat", "lon"], data_tuple[1])),
            coords=dict(lon=grids_dict['x_grid'], lat=grids_dict['y_grid'],
                        time=np.round(np.array(grids_dict['t_grid']) * 1000, 0).astype('datetime64[ms]')))

    def map_analytical_function_over_area(self, grids_dict: dict) -> Tuple:
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
            Args:
              grids_dict: containing grids of x, y, t dimension
            Returns:
              data_tuple     containing the data in tuple format as numpy array (not yet in xarray form)
            """

        # Step 1: Create the meshgrid numpy matrices for each coordinate
        LAT, TIMES, LON = np.meshgrid(grids_dict['y_grid'], grids_dict['t_grid'], grids_dict['x_grid'])

        # Step 2: Feed the arrays into the solar radiation function and return the np.array
        u_data = self.u_current_analytical(lon=LON, lat=LAT, posix_time=TIMES)
        v_data = self.v_current_analytical(lon=LON, lat=LAT, posix_time=TIMES)

        return (u_data, v_data)

    def get_data_at_point(self, point: List[float], time: datetime) -> xr:
        """Function to get the data at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          xr object that is then processed by the respective data source for its purpose
          """
        return OceanCurrentVector(u=self.u_current_analytical(lon=point[0], lat=point[1], posix_time=time.timestamp()),
                                  v=self.v_current_analytical(lon=point[0], lat=point[1], posix_time=time.timestamp()))

    def viz_field(self, inside: Optional[bool] = True):
        """Visualization function for the currents."""
        # Step 1: get default subset from analytical
        if inside:
            x_range = [0, 2]
            y_range = [0, 1]
        else:
            x_range = [-5, 5]
            y_range = [-5, 5]
        xarray = self.get_data_over_area(x_interval=x_range, y_interval=y_range,
                                         t_interval=[datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc) for t in [0, 100]])
        grids_dict = self.get_grid_dict_from_xr(xarray)
        # Step 2: plot it
        ax = utils.plotting_utils.visualize_currents(0, grids_dict, xarray['water_u'].data, xarray['water_v'].data, autoscale=False, plot=False)
        ax.set_title("Analytical Current Field")
        plt.show()


### Actual implemented analytical Sources ###


class PeriodicDoubleGyre(OceanCurrentSourceAnalytical):
    """ The Periodic Double Gyre Analytical current Field.
    Note: the spatial domain is fixed to [0,2]x[0,1]
    Source: https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html#Sec7.1

    Args:
        source_config_dict: dict the key 'source_settings' to a dict with the relevant specific settings
                Beyond the settings of AnalyticalSource for the PeriodicDoubleGyre we need the keys:
                    v_amplitude:
                        float representing maximum current strength in space units/time units.
                    epsilon_sep:
                        float >= 0 representing the magnitude of oscillation of the gyre around x=1.
                        The flow becomes time-independent at epsilon_sep = 0
                    period_time:
                        positive float of a full period time of an oscillation in time units.
    """

    def __init__(self, source_config_dict):
        super().__init__(source_config_dict)

        self.v_amplitude = source_config_dict['source_settings']['v_amplitude']
        self.epsilon_sep = source_config_dict['source_settings']['epsilon_sep']
        self.period_time = source_config_dict['source_settings']['period_time']

    def u_current_analytical(self, lon: Union[float, np.array], lat: Union[float, np.array],
                             posix_time: Union[float, np.array]) -> Union[float, np.array]:
        """ Analytical Formula for u velocity of Periodic Double Gyre.
            Note: this can be used for floats as input or np.arrays which are all the same shape.
            Args:
                lon: longitude in degree
                lat: latitude in degree
                posix_time: POSIX time
            Returns:
                u_currents     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
            """
        w_angular_vel = 2 * np.pi / self.period_time
        a = self.epsilon_sep * np.sin(w_angular_vel * posix_time)
        b = 1 - 2 * self.epsilon_sep * np.sin(w_angular_vel * posix_time)
        f = a * np.power(a * lon, 2) + b * lon

        u_cur_out = -np.pi * self.v_amplitude * np.sin(np.pi * f) * np.cos(np.pi * lat)
        return np.where(self.is_boundary(lon=lon, lat=lat, posix_time=posix_time), 0., u_cur_out)

    def v_current_analytical(self, lon: Union[float, np.array], lat: Union[float, np.array],
                             posix_time: Union[float, np.array]) -> Union[float, np.array]:
        """ Analytical Formula for v velocity of Periodic Double Gyre.
            Note: this can be used for floats as input or np.arrays which are all the same shape.
            Args:
                lon: longitude in degree
                lat: latitude in degree
                posix_time: POSIX time
            Returns:
                v_currents     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
            """
        w_angular_vel = 2 * np.pi / self.period_time
        a = self.epsilon_sep * np.sin(w_angular_vel * posix_time)
        b = 1 - 2 * self.epsilon_sep * np.sin(w_angular_vel * posix_time)
        f = a * np.power(a * lon, 2) + b * lon
        df_dx = 2 * a * lon + b

        v_cur_out = np.pi * self.v_amplitude * np.cos(np.pi * f) * np.sin(np.pi * lat) * df_dx
        return np.where(self.is_boundary(lon=lon, lat=lat, posix_time=posix_time), 0., v_cur_out)


class FixedCurrentHighwayField(OceanCurrentSourceAnalytical):
    """ The Highway current Field.

        Args:
        source_config_dict: dict the key 'source_settings' to a dict with the relevant specific settings
                Beyond the settings of AnalyticalSource for the PeriodicDoubleGyre we need the keys:
                    y_range_highway:
                        list representing the y-axis range of the highway current e.g. [3, 5]
                    U_cur:
                        strength of the current in space units/ time unit

        Example:
            ocean_source_dict = {
                'field': 'OceanCurrents',
                'source': 'analytical',
                'source_settings': {
                    'name': 'FixedCurrentHighwayField',
                    'boundary_buffers': [0.2, 0.2],
                    'x_domain': [0, 10],
                    'y_domain': [0, 10],
                    'temporal_domain': [0, 10],
                    'spatial_resolution': 0.1,
                    'temporal_resolution': 1,
                    'y_range_highway': [4,6],
                    'U_cur': 2,
                },
            }
        """

    def __init__(self, source_config_dict):
        super().__init__(source_config_dict)

        self.y_range_highway = source_config_dict['source_settings']['y_range_highway']
        self.U_cur = source_config_dict['source_settings']['U_cur']

    def u_current_analytical(self, lon: Union[float, np.array], lat: Union[float, np.array],
                             posix_time: Union[float, np.array]) -> Union[float, np.array]:
        """ Analytical Formula for u velocity.
            Note: this can be used for floats as input or np.arrays which are all the same shape.
            Args:
                lon: longitude in degree
                lat: latitude in degree
                posix_time: POSIX time
            Returns:
                v_currents     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
            """
        u_cur_low = np.where(lat <= self.y_range_highway[1], self.U_cur, 0.)
        u_cur_out = np.where(self.y_range_highway[0] <= lat, u_cur_low, 0.)
        return u_cur_out

    def v_current_analytical(self, lon: Union[float, np.array], lat: Union[float, np.array],
                             posix_time: Union[float, np.array]) -> Union[float, np.array]:
        """ Analytical Formula for v velocity.
            Note: this can be used for floats as input or np.arrays which are all the same shape.
            Args:
                lon: longitude in degree
                lat: latitude in degree
                posix_time: POSIX time
            Returns:
                v_currents     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
            """
        return np.zeros(lon.shape)
