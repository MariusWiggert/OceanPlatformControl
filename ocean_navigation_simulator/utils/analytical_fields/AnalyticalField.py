import datetime

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from functools import partial
import jax
from ocean_navigation_simulator import utils


class AnalyticalField:

    def __init__(self, spatial_domain, spatial_output_shape, temporal_domain, temporal_default_length):
        self.spatial_domain = hj.sets.Box(lo=spatial_domain[0], hi=spatial_domain[1])
        self.temporal_domain = hj.sets.Box(lo=np.array([temporal_domain[0]]),
                                           hi=np.array([temporal_domain[1]]))
        self.spatial_output_shape = spatial_output_shape
        self.temporal_output_shape = temporal_default_length

        # Quick gut check
        if not (self.spatial_domain.ndim == len(spatial_output_shape) == 2):
            raise ValueError("spatial_output_shape does not fit in dimension with spatial_domain")
        if not (self.temporal_domain.ndim == 1 and isinstance(temporal_default_length, int)):
            raise ValueError("temporal_domain and temporal_output_shape must be 1D")

    def u_current_analytical(self, state, time):
        """To be implemented in the child class. Note only for 2D currently."""
        print("TBD, u_current_analytical")

    def v_current_analytical(self, state, time):
        """To be implemented in the child class. Note only for 2D currently."""
        print("TBD, v_current_analytical")

    def get_grid_dict(self, t_interval=None, lat_interval=None, lon_interval=None, spatial_shape=None, temporal_res=None):
        """Helper Function to produce a grid dict."""

        # Step 1: Check default or requested shape
        if spatial_shape is None:
            spatial_shape = self.spatial_output_shape
        if temporal_res is None:
            temporal_res = self.temporal_output_shape
        if t_interval is None:
            t_interval = [self.temporal_domain.lo, self.temporal_domain.hi]
        if lon_interval is None:
            lon_interval = [self.spatial_domain.lo[0], self.spatial_domain.hi[0]]
        if lat_interval is None:
            lat_interval = [self.spatial_domain.lo[1], self.spatial_domain.hi[1]]

        # Step 1: Calculate the coordinate vectors and spatial meshgrid
        lo_hi_vec = [[lon, lat] for lon, lat in zip(lon_interval, lat_interval)]
        temporal_vector = jnp.linspace(t_interval[0], t_interval[1], temporal_res)
        coordinate_vectors = [jnp.linspace(l, h, n) for l, h, n in zip(lo_hi_vec[0], lo_hi_vec[1], spatial_shape)]

        grid_dict = {"t_range": t_interval,
                     "y_range": lat_interval, "x_range": lon_interval,
                     'x_grid': coordinate_vectors[0], 'y_grid': coordinate_vectors[1], 't_grid': temporal_vector.flatten(),
                      'spatial_land_mask': np.broadcast_to(False, spatial_shape).T}
        return grid_dict, coordinate_vectors

    def get_subset_from_analytical_field(self, t_interval, lat_interval, lon_interval,
                                         spatial_shape=None, temporal_res=None):
        """Returns u_data, v_data, and grid_dict for the requested subset.
        - grids_dict              dict containing x_grid, y_grid, t_grid, and spatial_land_mask (2D X,Y array)
        - u_data                  [T, Y, X] matrix of the ocean currents in x direction in m/s
        - v_data                  [T, Y, X] matrix of the ocean currents in y direction in m/s
        """
        # Step 0.0: if t_interval is in datetime convert to relative time
        if isinstance(t_interval[0], datetime.datetime):
            t_interval = [time.timestamp() for time in t_interval]
        # Step 0: check if within the spatial and temporal domain. Otherwise modify.
        lat_interval = [max(lat_interval[0], self.spatial_domain.lo[1]),
                        min(lat_interval[1], self.spatial_domain.hi[1])]
        lon_interval = [max(lon_interval[0], self.spatial_domain.lo[0]),
                        min(lon_interval[1], self.spatial_domain.hi[0])]

        t_interval = [max(t_interval[0], self.temporal_domain.lo[0]), min(t_interval[1], self.temporal_domain.hi[0])]

        # get the grid dict and coordinate_vectors
        grid_dict, coordinate_vectors= self.get_grid_dict(t_interval, lat_interval, lon_interval,
                                                           spatial_shape, temporal_res)

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

    def viz_field(self, inside=True):
        # Step 1: get default subset from analytical
        if inside:
            x_range = [0,2]
            y_range = [0,1]
        else:
            x_range = [-5, 5]
            y_range = [-5, 5]
        grid_dict, u_data, v_data = self.get_subset_from_analytical_field([0, 100], y_range, x_range)
        # Step 2: plot it
        ax = utils.plotting_utils.visualize_currents(0, grid_dict, u_data, v_data, autoscale=False, plot=False)
        ax.set_title("Analytical Current Field")
        plt.show()
