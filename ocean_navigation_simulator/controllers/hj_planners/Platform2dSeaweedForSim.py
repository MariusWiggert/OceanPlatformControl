from typing import Union

import jax.numpy as jnp
import xarray as xr
from hj_reachability import interpolation, sets

from ocean_navigation_simulator.controllers.hj_planners.Platform2dForSim import (
    Platform2dForSim,
    Platform2dForSimAffine
)


class Platform2dSeaweedForSim:
    """The 2D Ocean going Platform class on a dynamic current field considering seaweed growth.
    This class is for use with the ocean_platform simulator

    Dynamics:
    dot{x}_1 = u*u_max*cos(alpha) + x_currents(x,y,t)
    dot{x}_2 = u*u_max*sin(alpha) + y_currents(x,y,t)
    seaweed_growth -> check growth model
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
        affine_dynamics: bool = False,
    ):
        self.base_sim_class = Platform2dForSimAffine(
            u_max, d_max, use_geographic_coordinate_system, control_mode, disturbance_mode
        ) if affine_dynamics else Platform2dForSim(
            u_max, d_max, use_geographic_coordinate_system, control_mode, disturbance_mode
        )

    # called when an attribute is not found (for flexible inheritance)
    def __getattr__(self, name):
        return self.base_sim_class.__getattribute__(name)

    def update_jax_interpolant_seaweed(self, seaweed_xarray: xr):
        """Creating an interpolant function from x,y,t grid and data
        Args:
            seaweed_xarray: xarray containing variables F_NGR_per_second as matrices (T, Y, X)
        """
        # create 1D interpolation functions for running in the loop of the dynamics
        self.seaweed_rate = lambda state, time: interpolation.lin_interpo_1D(
            state,
            time,
            seaweed_xarray["F_NGR_per_second"].fillna(0).data,
            seaweed_xarray["lon"].data,
            seaweed_xarray["lat"].data,
            seaweed_xarray["relative_time"].data,
        )

    def _get_seaweed_growth_rate(self, state, time):
        return self.seaweed_rate(state, time)  # self._get_dim_state(state)

    def _get_dim_state(self, state_nonDim: jnp.ndarray):
        """Returns the state transformed from non_dimensional coordinates to dimensional coordinates."""
        return state_nonDim * self.characteristic_vec + self.offset_vec

    def hamiltonian(self, state, time, value, grad_value):
        """Evaluates the HJ PDE Hamiltonian and adds running cost term (negative seaweed growth rate)"""
        del value  # unused
        control, disturbance = self.optimal_control_and_disturbance(state, time, grad_value)
        return grad_value @ self.base_sim_class(state, control, disturbance, time) - self._get_seaweed_growth_rate(
            state, time
        )


class Platform2dSeaweedForSimDiscount(Platform2dSeaweedForSim):
    """Only difference to above is that we use a discount factor tau in the hamiltonian.
    The new hamiltonian is: -dV/dt = max_u [l + dV/dx*f] - V/tau

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
            discount_factor_tau: float = 1.0,
            affine_dynamics: bool = False,
    ):
        super().__init__(
            u_max, d_max, use_geographic_coordinate_system, control_mode, disturbance_mode,
            affine_dynamics=affine_dynamics
        )
        self.discount_factor_tau = discount_factor_tau

    def hamiltonian(self, state, time, value, grad_value):
        """Evaluates the HJ PDE Hamiltonian and adds running cost term (negative seaweed growth rate)"""
        control, disturbance = self.optimal_control_and_disturbance(state, time, grad_value)

        grad_term = grad_value @ self.base_sim_class(state, control, disturbance, time)
        running_cost_term = -self._get_seaweed_growth_rate(state, time)

        return grad_term + running_cost_term - value / self.discount_factor_tau