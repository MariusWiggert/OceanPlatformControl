from typing import Union

import jax.numpy as jnp
import xarray as xr
from pathlib import Path
import os

from ocean_navigation_simulator.controllers.hj_planners.Platform2dForSim import (
    Platform2dForSim, Platform2dForSimAffine
)


class Platform2dObsForSim(Platform2dForSimAffine):
    """The 2D Ocean going Platform class on a dynamic current field with obstacles.
    This class is for use with the ocean_platform simulator

    Dynamics:
    dot{x}_1 = u*u_max*cos(alpha) + x_currents(x,y,t)
    dot{x}_2 = u*u_max*sin(alpha) + y_currents(x,y,t)
    such that u in [0,1] and alpha in [0, 2pi]
    The controls are u and alpha.

    Args:
        u_max: the maximum propulsion in m/s
        d_max: the maximum disturbance in m/s (default is 0)
        use_geographic_coordinate_system: if we operate in the geographic coordinate system or not
        control_mode: If the control is trying to minimize or maximize the value function.
        disturbance_mode: If the disturbance is trying to minimize or maximize the value function.
        obstacle_file: Name of the xarray file containing the distance to the obstacle (in package_data/bathymetry_and_garbage/).
        safe_distance_to_obstacle: Use to overapproximate obstacles by value to ensure whole obstacle is masked.
    """

    def __init__(
        self,
        u_max: float,
        d_max: float = 0,
        use_geographic_coordinate_system: bool = True,
        control_mode: Union["min", "max"] = "min",
        disturbance_mode: Union["min", "max"] = "max",
        obstacle_file: str = None,
        safe_distance_to_obstacle: float = 0,
    ):
        super().__init__(
            u_max, d_max, use_geographic_coordinate_system, control_mode, disturbance_mode
        )
        # make it an absolute path to the obstacle file
        self.path_to_obstacle_file = os.path.join(Path(__file__).resolve().parents[2],
                                        'package_data/bathymetry_and_garbage/',
                                        obstacle_file)

        self.safe_distance_to_obstacle = safe_distance_to_obstacle

    def update_jax_interpolant(self, data_xarray: xr):
        """Creating an interpolant function from x,y,t grid and data
        Args:
            data_xarray: xarray containing variables water_u and water_v as matrices (T, Y, X)
        """
        super().update_jax_interpolant(data_xarray)
        self.binary_obs_map, self.obs_x_axis, self.obs_y_axis = self.create_obstacle_arrays(data_xarray)

    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE with obstacles."""
        dx_out = super().__call__(state, control, disturbance, time)
        # Check if state is in obstacle then mask dx_out,
        # this means that the "state moves with obstacle" and can not leave it.
        # Additionally, we mask the obstacles in the hamiltonian_postprocessor in "HJPLannerBase"
        # to counter the dissipation value of the artificial_dissipation_scheme
        # TODO: Would actually need to change the 0 to velocity of obstacle if obstacle were dynamic
        return jnp.where(self.is_in_obstacle(state, time), 0, dx_out)

    def create_obstacle_arrays(self, data_xarray):
        """Use path to file to load and set the obstacle array"""
        obstacle_ds = xr.open_dataset(self.path_to_obstacle_file)["distance"]
        # Fit to the grid that is used for HJ computation to have same resolution
        obstacle_array = obstacle_ds.interp_like(data_xarray)
        # check if part of it is out of range (obstacle_array contains NaN), if yes through error
        if obstacle_array.isnull().any():
            raise ValueError("The obstacle file does not cover the whole domain."
                             "Reduce the domain or update obstacle file."
                             "\n Data Area Domain: {}."
                             "\n Obstacle File Domain: {}."
                             "\n File {}).".format(data_xarray.coords, obstacle_ds.coords, self.path_to_obstacle_file))
        # Convert to binary mask, set to 0 for "no obstacle" and 1 for "obstacle"
        binary_obs_ds = xr.where(obstacle_array > self.safe_distance_to_obstacle, 0, 1)
        # return all as jnp arrays
        return jnp.array(binary_obs_ds.data), jnp.array(binary_obs_ds['lon'].data), jnp.array(binary_obs_ds['lat'].data)

    def is_in_obstacle(self, state, time):
        """Return if the state is in the obstacle region"""
        x_idx = jnp.argmin(jnp.abs(self.obs_x_axis - state[0]))
        y_idx = jnp.argmin(jnp.abs(self.obs_y_axis - state[1]))
        return self.binary_obs_map[y_idx, x_idx] > 0
