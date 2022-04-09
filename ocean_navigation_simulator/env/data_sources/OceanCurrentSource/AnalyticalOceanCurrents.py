import datetime
import matplotlib.pyplot as plt
from typing import List, NamedTuple, Sequence, AnyStr, Optional, Tuple
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
    def u_current_analytical(self, point: List[float], time: float):
        """To be implemented in the child class. Note only for 2D currently."""
        raise NotImplementedError

    @abc.abstractmethod
    def v_current_analytical(self, point: List[float], time: float):
        """To be implemented in the child class. Note only for 2D currently."""
        raise NotImplementedError

    def create_xarray(self, grid_dict: dict, data_tuple: Tuple) -> xr:
        """Function to create an xarray from the data tuple and grid dict
            Args:
              data_tuple: tuple containing (data_u, data_v)
              grid_dict: containing ranges and grids of x, y, t dimension
            Returns:
              xr     an xarray containing both the grid and data
            """
        # make a xarray object out of it
        return xr.Dataset(
            dict(water_u=(["time", "lat", "lon"], data_tuple[0]), water_v=(["time", "lat", "lon"], data_tuple[1])),
            coords=dict(lon=grid_dict['x_grid'], lat=grid_dict['y_grid'],
                        time=np.round(np.array(grid_dict['t_grid']) * 1000, 0).astype('datetime64[ms]')))

    def map_analytical_function_over_area(self, states, grid_dict) -> Tuple:
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
            Args:
              states: jax.numpy array containing the desired spatial grid as states
              grid_dict: containing ranges and grids of x, y, t dimension
            Returns:
              data_tuple     containing the data in tuple format as numpy array (not yet in xarray form)
            """
        # Step 2: Map analytical function over the area
        u_func_over_space = lambda t: hj.utils.multivmap(partial(self.u_current_analytical, time=t), np.arange(2))(
            jnp.transpose(states, axes=[1, 0, 2]))
        v_func_over_space = lambda t: hj.utils.multivmap(partial(self.v_current_analytical, time=t), np.arange(2))(
            jnp.transpose(states, axes=[1, 0, 2]))
        # execute over temporal_vector
        u_data = jax.vmap(u_func_over_space)(grid_dict['t_grid'])
        v_data = jax.vmap(v_func_over_space)(grid_dict['t_grid'])
        return (u_data, v_data)

    def get_data_at_point(self, point: List[float], time: datetime) -> xr:
        """Function to get the data at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          xr object that is then processed by the respective data source for its purpose
          """
        return OceanCurrentVector(u=self.u_current_analytical(point, time.timestamp()),
                                  v=self.v_current_analytical(point, time.timestamp()))

    def get_time_relative_to_t_0(self, time):
        """Helper function because with non-dimensionalization we run the u and v currents in relative time."""
        return time - self.current_run_t_0

    def viz_field(self, inside: Optional[bool] = True):
        """Visualization function for the currents."""
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


class FixedCurrentHighwayField(OceanCurrentSourceAnalytical):
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
