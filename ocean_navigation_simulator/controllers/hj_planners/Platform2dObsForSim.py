from typing import Union
import jax.numpy as jnp
import xarray as xr
from ocean_navigation_simulator.controllers.hj_planners.Platform2dForSim import Platform2dForSim


class Platform2dObsForSim(Platform2dForSim):
    """The 2D Ocean going Platform class on a dynamic current field.
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
    """

    def __init__(
        self,
        u_max: float,
        d_max: float = 0,
        use_geographic_coordinate_system: bool = True,
        control_mode: Union["min", "max"] = "min",
        disturbance_mode: Union["min", "max"] = "max",
        path_to_obstacle_file: str = None,
    ):
        # initialize the parent class
        super().__init__(u_max, d_max, use_geographic_coordinate_system, control_mode, disturbance_mode)
        # set obstacle array
        self.path_to_obstacle_file = path_to_obstacle_file
        self.obstacle_array = None

    def update_jax_interpolant(self, data_xarray: xr):
        """Creating an interpolant function from x,y,t grid and data
        Args:
            data_xarray: xarray containing variables water_u and water_v as matrices (T, Y, X)
        """
        # call parent function
        super().update_jax_interpolant(data_xarray)
        # use bounds of xarray to create obstacle array
        self.obstacle_array = self.create_obstacle_array(data_xarray)

    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE."""
        dx_out = super.__call__(state, control, disturbance, time)
        # check if state is in obstacle then mask dx_out
        return jnp.where(self.check_if_in_obstacle(state, time), 0, dx_out)

    def create_obstacle_array(self, data_xarray):
        """Use path to file to load and set the obstacle array"""
        pass

    def check_if_in_obstacle(self, state, time):
        """Check if the state is in the obstacle region and return boolean"""
        pass
