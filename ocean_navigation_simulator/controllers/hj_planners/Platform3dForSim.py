import jax.lax
import jax.numpy as jnp
from hj_reachability import dynamics, sets, interpolation
from typing import AnyStr, Union
import xarray as xr


def transform_to_geographic_velocity(state, dx1, dx2):
    """Helper Function to transform dx1 and dx2 from m/s to the geographic_coordinate_system."""
    lon_delta_deg_per_s = 180 * dx1 / jnp.pi / 6371000 / jnp.cos(jnp.pi * state[1] / 180)
    lat_delta_deg_per_s = 180 * dx2 / jnp.pi / 6371000
    return jnp.array([lon_delta_deg_per_s, lat_delta_deg_per_s]).reshape(-1)


class Platform3dForSim(dynamics.Dynamics):
    """ The 3D Ocean going Platform class on a dynamic current field with with 2D space & 1D energy dimension.
    This class is for use with the ocean_platform simulator
    Dynamics:
    dot{x}_1 = u*u_max*cos(alpha) + x_currents(x,y,t)
    dot{x}_2 = u*u_max*sin(alpha) + y_currents(x,y,t)
    dot{x}_3 = (c - D * (u_max * u)^3)             # Note: potentially *I(b in [0,1])
    such that u in [0,1] and alpha in [0, 2pi]
    The controls are u and alpha.

    Args:
        u_max: the maximum propulsion in m/s
        d_max: the maximum disturbance in m/s (default is 0)
        use_geographic_coordinate_system: if we operate in the geographic coordinate system or not
        control_mode: If the control is trying to minimize or maximize the value function.
        disturbance_mode: If the disturbance is trying to minimize or maximize the value function.
        c: solar charge rate in W
        D: drag factor of flow performance in kg/m D = 0.5 * density_water * coeff_drag * A_seaweed 
    """

    def __init__(self, u_max: float, d_max: float = 0,
                 use_geographic_coordinate_system: bool = True,
                 control_mode: Union["min", "max"] = "min",
                 disturbance_mode: Union["min", "max"] = "max", 
                 c=0., D=675.): # TODO: check default value for D (in Notion it is estimated as 675 kg/ms)

        # set variables
        self.u_max = jnp.array(u_max)
        self.use_geographic_coordinate_system = use_geographic_coordinate_system
        self.c = jnp.array(c)
        self.D = jnp.array(D)

        # initialize the current interpolants with None, they are set in the planner method
        self.x_current, self.y_current = None, None

        # # obstacle operator (is overwritten if analytical_current with boundary obstacles)
        # self.obstacle_operator = lambda state, time, dx_out: dx_out

        control_space = sets.Box(lo=jnp.array([0, 0]),
                                 hi=jnp.array([1., 2 * jnp.pi]))

        disturbance_space = sets.Ball(center=jnp.zeros(2), radius=d_max)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def update_jax_interpolant(self, data_xarray: xr):
        """Creating an interpolant function from x,y,t grid and data
        Args:
            data_xarray: xarray containing variables water_u and water_v as matrices (T, Y, X)
        """

        # create 1D interpolation functions for running in the loop of the dynamics
        self.x_current = lambda state, time: interpolation.lin_interpo_1D(state, time,
                                                                          data_xarray['water_u'].fillna(0).data,
                                                                          data_xarray['lon'].data,
                                                                          data_xarray['lat'].data,
                                                                          data_xarray['relative_time'].data)
        self.y_current = lambda state, time: interpolation.lin_interpo_1D(state, time,
                                                                          data_xarray['water_v'].fillna(0).data,
                                                                          data_xarray['lon'].data,
                                                                          data_xarray['lat'].data,
                                                                          data_xarray['relative_time'].data)

    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE."""
        # dx is in m/s
        dx1 = self.u_max * control[0] * jnp.cos(control[1]) + self.x_current(state, time) + disturbance[0]
        dx2 = self.u_max * control[0] * jnp.sin(control[1]) + self.y_current(state, time) + disturbance[1]

        # dx is in W
        dx3 = self.c - self.D * jnp.power((self.u_max * control[0]), 3)

        # now transfer it to deg lat/lon per second if use_geographic_coordinate_system        
        if self.use_geographic_coordinate_system:
            geometric_velocity = transform_to_geographic_velocity(state, dx1, dx2)
            dx_out = jnp.append(geometric_velocity, dx3)
        else:
            dx_out = jnp.array([dx1, dx2, dx3]).reshape(-1)

        return dx_out
        # return self.obstacle_operator(state, time, dx_out)

    @staticmethod
    def disturbance_jacobian(state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.]
        ])

    def optimal_control(self, state, time, grad_value):
        """Computes the optimal control realized by the HJ PDE Hamiltonian."""
        uOpt = jnp.array(1.)
        # angle of px, py vector of gradient
        alpha = jax.lax.atan2(grad_value[1], grad_value[0])
        # if min, go against the gradient direction
        if self.control_mode == 'min':
            alpha = alpha + jnp.pi

        # calculate u_mid
        a = jnp.linalg.norm(grad_value[:2])
        b = jnp.abs(grad_value[2]) * self.D * jnp.power(self.u_max, 2)
        u_mid = jnp.power(jnp.divide(a, 3. * b), 0.5)

        uOpt = jnp.where(grad_value[2] >= 0, x=1., y=jnp.minimum(u_mid, 1.))

        return jnp.array([uOpt, alpha])

    def optimal_disturbance(self, state, time, grad_value):
        """Computes the optimal disturbance realized by the HJ PDE Hamiltonian."""
        disturbance_direction = grad_value[:2] @ self.disturbance_jacobian(state, time)
        return self.disturbance_space.extreme_point(disturbance_direction)

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        if self.control_mode == "max" or self.disturbance_mode == "min":
            raise ImportError("The max, min option is not yet implemented.")
        return (self.optimal_control(state, time, grad_value), self.optimal_disturbance(state, time, grad_value))