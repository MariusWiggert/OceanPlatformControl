import jax.lax
import jax.numpy as jnp
from hj_reachability import dynamics, sets, interpolation


class Platform3D_for_sim(dynamics.Dynamics):
    """ The 3D Ocean going Platform class with 2D space & 1D energy dimension.
    This class is for use with the ocean_platform simulator.

    Dimensional Dynamics:
    dot{x}_1 = u_max*u*cos(alpha) + x_currents(x,y,t)
    dot{x}_2 = u_max*u*sin(alpha) + y_currents(x,y,t)
    dot{x}_3 = (c - D * (u_max * u)^3                  # Note: with *I(b in [0,1])
    with the controls u in [0, 1] and alpha in [0, 2pi]

    Input Params:
    - u_max             in m/s
    - c                 average relative charging of the battery in 1/s
    - D                 drag factor of the platform which determines relative energy usage D*u^3
    - space_coeff       a factor used to multiply the dynamics (e.g. used to transform m/s to deg/s)
    """

    def __init__(self, u_max, c, D,
                 space_coeff=1./111120.,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        # set variables
        self.u_max = jnp.array(u_max)
        self.c = jnp.array(c)
        self.D = jnp.array(D)
        self.space_coeff = space_coeff

        # initialize the current interpolants with None, they are set in the planner method
        self.x_current, self.y_current = None, None

        if control_space is None:
            control_space = sets.Box(lo=jnp.array([0, 0]),
                                     hi=jnp.array([1., 2 * jnp.pi]))
        if disturbance_space is None:
            disturbance_space = sets.Ball(center=jnp.zeros(2), radius=0.)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def update_jax_interpolant(self, x_grid, y_grid, t_grid, water_u, water_v):
        """Creating an interpolant function from x,y,t grid and data
        Input Params:
        - water_u, water_v      the array comes in from HYCOM as (T, Y, X)
        """
        # need to flip array around to fit with grid_axis and dynamics
        water_u = jnp.swapaxes(water_u, 0, 2)
        water_v = jnp.swapaxes(water_v, 0, 2)
        self.x_current = lambda point: interpolation.lin_interpo_3D_fields(
            point, water_u, x_grid, y_grid, jnp.array(t_grid))
        self.y_current = lambda point: interpolation.lin_interpo_3D_fields(
            point, water_v, x_grid, y_grid, jnp.array(t_grid))

    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE."""
        # TODO: next iteration with (lat, lon, time) varying solar radiation field/forecasts etc.
        # calculation happens in m/s
        dx1 = self.u_max * control[0] * jnp.cos(control[1]) + self.x_current(jnp.append(state, time))
        dx2 = self.u_max * control[0] * jnp.sin(control[1]) + self.y_current(jnp.append(state, time))
        dx3 = self.c - self.D * jnp.power(self.u_max * control[0], 3)
        # jnp.where, and able to decrease (and other way round...)
        # potentially have different function for forward/backwards (let's see)
        # How can we put the solar-radiation field into this? via nc_file or jit the function somehow..
        # Do current interpolation outside of the loop?
        # units get transformed by space_coeff
        return jnp.array([self.space_coeff * dx1, self.space_coeff * dx2, dx3])

    @staticmethod
    def disturbance_jacobian(state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.]
        ])

    def optimal_control(self, state, time, grad_value):
        """Computes the optimal control realized by the HJ PDE Hamiltonian."""
        # ToDo: subdivide bases on the min vs the max controller setting
        # angle of px, py vector of gradient
        alpha = jax.lax.atan2(grad_value[1], grad_value[0]) + jnp.pi

        # calculate u_mid
        a = jnp.linalg.norm(grad_value[:2])
        b = jnp.abs(grad_value[2]) * self.D * jnp.power(self.u_max, 2)
        u_mid = jnp.power(jnp.divide(a, 3. * b), 0.5)

        uOpt = jnp.where(grad_value[2] >= 0, x=1., y=jnp.minimum(u_mid, 1.))

        return jnp.array([uOpt, alpha])

    def optimal_disturbance(self, state, time, grad_value):
        """Computes the optimal disturbance realized by the HJ PDE Hamiltonian."""
        disturbance_direction = grad_value[:2] @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        return self.disturbance_space.extreme_point(disturbance_direction)

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        return (self.optimal_control(state, time, grad_value),
                self.optimal_disturbance(state, time, grad_value))