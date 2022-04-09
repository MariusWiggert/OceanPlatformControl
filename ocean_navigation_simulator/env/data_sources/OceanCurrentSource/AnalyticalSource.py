import datetime
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from typing import List, NamedTuple, Sequence, AnyStr, Optional
import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from functools import partial
import xarray as xr
import jax
from ocean_navigation_simulator import utils
from ocean_navigation_simulator.env.data_sources.DataField import DataField
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
import abc


# TODO: How does the direction connection of the analytical function into HJ Planner work? Is it even desirable?
# -> it's possible but I would have to give up passing OceanCurrentVectors around but rather the numbers directly.


class AnalyticalSource(OceanCurrentSource):

    def __init__(self, source_config_dict: dict):
        """Class for Analytical Ocean Current Sources
            Args:
              source_config_dict: dict the key 'source_settings' to a dict with the relevant specific settings
                The general AnalyticalSource requires the following keys, the explicit analytical currents some extra.
                    boundary_buffers:
                            Margin to buffer the spatial domain with obstacles as boundary conditions e.g. [0.2, 0.2]
                    spatial_domain:
                            a list e..g [np.array([-0.1, -0.1]), np.array([2.1, 1.1])],
                    temporal_domain:
                            a list e.g. [-10, 1000] of the temporal domain in units (will internally be seconds)
                    spatial_resolution:
                            a float as the default spatial_resolution
                    temporal_default_length:
                            an int of the default length of the time dimension when called
        """
        super().__init__(source_config_dict)
        # Step 1: Some basic initializations
        # adjust spatial domain by boundary buffer
        self.boundary_buffers = np.array(source_config_dict['source_settings']['boundary_buffers'])
        spatial_domain = [source_config_dict['source_settings']['spatial_domain'][0] - self.boundary_buffers,
                          source_config_dict['source_settings']['spatial_domain'][1] + self.boundary_buffers]

        self.spatial_domain = hj.sets.Box(lo=spatial_domain[0],
                                          hi=spatial_domain[1])
        self.temporal_domain = hj.sets.Box(lo=np.array([source_config_dict['source_settings']['temporal_domain'][0]]),
                                           hi=np.array([source_config_dict['source_settings']['temporal_domain'][1]]))

        self.spatial_resolution = source_config_dict['source_settings']['spatial_resolution']
        self.temporal_resolution = source_config_dict['source_settings']['temporal_resolution']
        self.current_run_t_0 = 0.

        # Quick gut check
        if not self.temporal_domain.ndim == 1:
            raise ValueError("temporal_domain  must be 1D")

        # Step 3: derive a general grid_dict
        self.grid_dict = self.get_grid_dict()

    @abc.abstractmethod
    def u_current_analytical(self, point: List[float], time: float):
        """To be implemented in the child class. Note only for 2D currently."""
        print("TBD, u_current_analytical")

    @abc.abstractmethod
    def v_current_analytical(self, point: List[float], time: float):
        """To be implemented in the child class. Note only for 2D currently."""
        print("TBD, v_current_analytical")

    def get_currents_at_point(self, point: List[float], time: datetime) -> OceanCurrentVector:
        """Function to get the OceanCurrentVector at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          OceanCurrentVector
          """

        return OceanCurrentVector(u=self.u_current_analytical(point, time.timestamp()),
                                  v=self.v_current_analytical(point, time.timestamp()))

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
          data_array     in xarray format that contains the grid and the values (boundary buffer here is NaN)
        """

        # Step 0.0: if t_interval is in datetime convert to relative time
        if isinstance(t_interval[0], datetime):
            t_interval_posix = [time.timestamp() for time in t_interval]
        else:
            t_interval_posix = t_interval

        grid_dict, u_data, v_data = self.get_subset_from_analytical_field(x_interval, y_interval, t_interval_posix,
                                                                          spatial_resolution, temporal_resolution)

        # make a xarray object out of it
        subset = xr.Dataset(
            dict(water_u=(["time", "lat", "lon"], u_data), water_v=(["time", "lat", "lon"], v_data)),
            coords=dict(lon=grid_dict['x_grid'], lat=grid_dict['y_grid'],
                        time=np.round(np.array(grid_dict['t_grid']) * 1000, 0).astype('datetime64[ms]')))

        # Step 3: Do a sanity check for the sub-setting before it's used outside and leads to errors
        DataField.array_subsetting_sanity_check(subset, x_interval, y_interval, t_interval)

        return subset

    def get_grid_dict(self, x_interval: Optional[List[float]] = None, y_interval: Optional[List[float]] = None,
                      t_interval: Optional[List[float]] = None,
                      spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None):
        """Helper Function to produce a grid dict."""

        # Step 1: Check default resolution or requested
        if spatial_resolution is None:
            spatial_resolution = self.spatial_resolution
        if temporal_resolution is None:
            temporal_resolution = self.temporal_resolution
        if t_interval is None:
            t_interval = [self.temporal_domain.lo, self.temporal_domain.hi]
        if x_interval is None:
            x_interval = [self.spatial_domain.lo[0], self.spatial_domain.hi[0]]
        if y_interval is None:
            y_interval = [self.spatial_domain.lo[1], self.spatial_domain.hi[1]]

        # Step 1: Calculate the coordinate vectors and spatial meshgrid
        lo_hi_vec = [[lon, lat] for lon, lat in zip(x_interval, y_interval)]
        temporal_vector = jnp.linspace(t_interval[0], t_interval[1], temporal_resolution)
        # The +0.01 is a hacky way to include the endpoint
        coordinate_vectors = [jnp.arange(start=l, stop=h+0.01, step=spatial_resolution) for l, h in zip(lo_hi_vec[0], lo_hi_vec[1])]

        grid_dict = {"t_range": t_interval,
                     "y_range": y_interval, "x_range": x_interval,
                     'x_grid': coordinate_vectors[0], 'y_grid': coordinate_vectors[1],
                     't_grid': temporal_vector.flatten()}
        return grid_dict, coordinate_vectors

    def get_subset_from_analytical_field(self, x_interval: List[float], y_interval: List[float],
                                         t_interval: List[float],
                                         spatial_resolution: Optional[float] = None, temporal_resolution: Optional[float] = None) -> xr:
        """Returns the xarray of the requested interval and resolution.
        """
        # Step 0: check if within the spatial and temporal domain. Otherwise modify.
        y_interval = [max(y_interval[0], self.spatial_domain.lo[1]),
                      min(y_interval[1], self.spatial_domain.hi[1])]
        x_interval = [max(x_interval[0], self.spatial_domain.lo[0]),
                      min(x_interval[1], self.spatial_domain.hi[0])]

        t_interval = [max(t_interval[0], self.temporal_domain.lo[0]), min(t_interval[1], self.temporal_domain.hi[0])]

        # get the grid dict and coordinate_vectors
        grid_dict, coordinate_vectors = self.get_grid_dict(x_interval, y_interval, t_interval,
                                                           spatial_resolution=spatial_resolution,
                                                           temporal_resolution=temporal_resolution)

        # Step 1: Calculate the coordinate vectors and spatial meshgrid
        states = jnp.stack(jnp.meshgrid(*coordinate_vectors, indexing="ij"), -1)

        # Step 2: Evaluate the analytical function over time on the spatial meshgrid
        u_func_over_space = lambda t: hj.utils.multivmap(partial(self.u_current_analytical, time=t), np.arange(2))(
            jnp.transpose(states, axes=[1, 0, 2]))
        v_func_over_space = lambda t: hj.utils.multivmap(partial(self.v_current_analytical, time=t), np.arange(2))(
            jnp.transpose(states, axes=[1, 0, 2]))
        # execute over temporal_vector
        u_data = jax.vmap(u_func_over_space)(grid_dict['t_grid'])
        v_data = jax.vmap(v_func_over_space)(grid_dict['t_grid'])

        return grid_dict, u_data, v_data

    def viz_field(self, inside: Optional[bool] = True):
        # Step 1: get default subset from analytical
        if inside:
            x_range = [0, 2]
            y_range = [0, 1]
        else:
            x_range = [-5, 5]
            y_range = [-5, 5]
        grid_dict, u_data, v_data = self.get_subset_from_analytical_field(x_interval=x_range,
                                                                          y_interval=y_range,
                                                                          t_interval=[0, 100])
        # Step 2: plot it
        ax = utils.plotting_utils.visualize_currents(0, grid_dict, u_data, v_data, autoscale=False, plot=False)
        ax.set_title("Analytical Current Field")
        plt.show()

    def get_time_relative_to_t_0(self, time):
        """Helper function because with non-dimensionalization we run the u and v currents in relative time."""
        return time - self.current_run_t_0

    def is_boundary(self, point: List[float], time: float):
        """Helper function to check if a state is in the boundary specified in hj as obstacle."""
        del time
        x_boundary = jnp.logical_or(point[0] < self.spatial_domain.lo[0] + self.boundary_buffers[0],
                                    point[0] > self.spatial_domain.hi[0] - self.boundary_buffers[0])
        y_boundary = jnp.logical_or(point[1] < self.spatial_domain.lo[1] + self.boundary_buffers[1],
                                    point[1] > self.spatial_domain.hi[1] - self.boundary_buffers[1])

        return jnp.logical_or(x_boundary, y_boundary)


### Actual implemented analytical Sources ###


class PeriodicDoubleGyre(AnalyticalSource):
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

    def u_current_analytical(self, point: List[float], time: float):
        """Analytical Formula for u velocity of Periodic Double Gyre."""
        time = self.get_time_relative_to_t_0(time)
        w_angular_vel = 2 * jnp.pi / self.period_time
        a = self.epsilon_sep * jnp.sin(w_angular_vel * time)
        b = 1 - 2 * self.epsilon_sep * jnp.sin(w_angular_vel * time)
        f = a * jnp.power(a * point[0], 2) + b * point[0]

        u_cur_out = -jnp.pi * self.v_amplitude * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * point[1])
        return jnp.where(self.is_boundary(point, time), 0., u_cur_out)

    def v_current_analytical(self, point: List[float], time: float):
        """Analytical Formula for u velocity of Periodic Double Gyre."""
        time = self.get_time_relative_to_t_0(time)
        w_angular_vel = 2 * jnp.pi / self.period_time
        a = self.epsilon_sep * jnp.sin(w_angular_vel * time)
        b = 1 - 2 * self.epsilon_sep * jnp.sin(w_angular_vel * time)
        f = a * jnp.power(a * point[0], 2) + b * point[0]
        df_dx = 2 * a * point[0] + b

        v_cur_out = jnp.pi * self.v_amplitude * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * point[1]) * df_dx
        return jnp.where(self.is_boundary(point, time), 0., v_cur_out)


class FixedCurrentHighwayField(AnalyticalSource):
    """ The Highway current Field.

        Args:
        source_config_dict: dict the key 'source_settings' to a dict with the relevant specific settings
                Beyond the settings of AnalyticalSource for the PeriodicDoubleGyre we need the keys:
                    y_range_highway:
                        list representing the y-axis range of the highway current e.g. [3, 5]
                    U_cur:
                        strength of the current in space units/ time unit
        """

    def __init__(self, source_config_dict):
        super().__init__(source_config_dict)

        self.y_range_highway = source_config_dict['source_settings']['y_range_highway']
        self.U_cur = source_config_dict['source_settings']['U_cur']

    def u_current_analytical(self, state, time):
        u_cur_low = jnp.where(state[1] <= self.y_range_highway[1], self.U_cur, 0.)
        u_cur_out = jnp.where(self.y_range_highway[0] <= state[1], u_cur_low, 0.)
        return u_cur_out

    def v_current_analytical(self, state, time):
        return 0.
