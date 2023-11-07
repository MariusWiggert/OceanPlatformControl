from typing import Union

import jax.numpy as jnp
import xarray as xr
from hj_reachability import interpolation, sets

from ocean_navigation_simulator.controllers.hj_planners.Platform2dForSim import Platform2dForSimAffine
from ocean_navigation_simulator.controllers.hj_planners.Platform2dObsForSim import Platform2dObsForSim


def seaweed_platform_factory(obstacles: bool = False, **kwargs):
    if obstacles:
        base_class = Platform2dObsForSim
    else:
        base_class = Platform2dForSimAffine

    class Platform2dSeaweedForSim(base_class):
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
            discount_factor_tau: how much to discount future growth versus near-term growth.
        """

        def __init__(
                self,
                u_max: float,
                d_max: float = 0,
                use_geographic_coordinate_system: bool = True,
                control_mode: Union["min", "max"] = "min",
                disturbance_mode: Union["min", "max"] = "max",
                **kwargs
                # obstacle_file: str = None,
                # safe_distance_to_obstacle: float = 0,
                # discount_factor_tau: float = 0,
        ):
            super().__init__(
                u_max, d_max, use_geographic_coordinate_system, control_mode, disturbance_mode, **kwargs)

            self.seaweed_rate = None
            self.discount_factor_tau = kwargs['discount_factor_tau'] if 'discount_factor_tau' in kwargs else 0

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
            return self.seaweed_rate(state, time)

        def hamiltonian(self, state, time, value, grad_value):
            """Evaluates the HJ PDE Hamiltonian and adds running cost term (negative seaweed growth rate).
            If a discount factor is given, then the hamiltonian changes to: -dV/dt = max_u [l + dV/dx*f] - V/tau
            """
            control, disturbance = self.optimal_control_and_disturbance(state, time, grad_value)

            grad_term = grad_value @ self(state, control, disturbance, time)
            running_cost_term = -self._get_seaweed_growth_rate(state, time)

            return grad_term + running_cost_term + jnp.where(self.discount_factor_tau, - value / self.discount_factor_tau, 0)

    return Platform2dSeaweedForSim(**kwargs)


