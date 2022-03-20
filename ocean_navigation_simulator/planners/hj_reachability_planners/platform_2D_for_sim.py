import jax.lax
import jax.numpy as jnp
from hj_reachability import dynamics, sets, interpolation

# TODO: Better way of fixing optimal disturbance to 0 as of right now it produces errors!


class Platform2D_for_sim(dynamics.Dynamics):
    """ The 2D Ocean going Platform class on a dynamic current field.
    This class is for use with the ocean_platform simulator

    Dynamics:
    dot{x}_1 = u*u_max*cos(alpha) + x_currents(x,y,t)
    dot{x}_2 = u*u_max*sin(alpha) + y_currents(x,y,t)
    such that u in [0,1] and alpha in [0, 2pi]
    The controls are u and alpha.

    Input Params:
    - u_max             in m/s
    - space_coeff       a factor used to multiply the dynamics (e.g. used to transform m/s to deg/s)
    """

    def __init__(self, u_max=1.,
                 space_coeff=1./111120.,
                 d_max=0,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        # set variables
        self.u_max = jnp.array(u_max)
        self.space_coeff = space_coeff

        # initialize the current interpolants with None, they are set in the planner method
        self.x_current, self.y_current = None, None

        if control_space is None:
            control_space = sets.Box(lo=jnp.array([0, 0]),
                                     hi=jnp.array([1., 2 * jnp.pi]))
        if disturbance_space is None:
            disturbance_space = sets.Ball(center=jnp.zeros(2), radius=d_max)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def update_jax_interpolant(self, x_grid, y_grid, t_grid, water_u, water_v):
        """Creating an interpolant function from x,y,t grid and data
        Input Params:
        - water_u, water_v      the array comes in from HYCOM as (T, Y, X)
        """

        # create 1D interpolation functions for running in the loop of the dynamics
        self.x_current = lambda point: interpolation.lin_interpo_1D(point, water_u, x_grid, y_grid, t_grid)
        self.y_current = lambda point: interpolation.lin_interpo_1D(point, water_v, x_grid, y_grid, t_grid)

    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE."""
        # calculation happens in m/s
        dx1 = self.u_max * control[0] * jnp.cos(control[1]) + self.x_current(jnp.append(state, time))
        dx2 = self.u_max * control[0] * jnp.sin(control[1]) + self.y_current(jnp.append(state, time))
        # units get transformed by space_coeff
        return self.space_coeff * jnp.array([dx1, dx2])

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
        return jnp.array([uOpt, alpha])

    def optimal_disturbance(self, state, time, grad_value):
        """Computes the optimal disturbance realized by the HJ PDE Hamiltonian."""
        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == 'min':
            disturbance_direction = -disturbance_direction
        # return self.disturbance_space.extreme_point(disturbance_direction)
        return jnp.zeros(2)

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        return (self.optimal_control(state, time, grad_value),
                self.optimal_disturbance(state, time, grad_value))


# In Multi-Reachability the inside is flat (so no gradients) that leads to problems when calculating the disturbance
# Hence, we need to split it in a class with disturbance and one without.
class Platform2D_for_sim_with_disturbance(Platform2D_for_sim):
    # Just overwrite the call and not add the disturbance term
    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE."""
        # calculation happens in m/s
        dx1 = self.u_max * control[0] * jnp.cos(control[1]) + self.x_current(jnp.append(state, time)) + disturbance[0]
        dx2 = self.u_max * control[0] * jnp.sin(control[1]) + self.y_current(jnp.append(state, time)) + disturbance[1]
        # units get transformed by space_coeff
        return self.space_coeff * jnp.array([dx1, dx2])

