from hj_reachability import dynamics, sets
import jax.numpy as jnp
import jax


class Platform2Dcurrents(dynamics.Dynamics):
    """ The 2D Ocean going Platform class on a dynamic current field.

    Dynamics:
    dot{x}_1 = u*u_max*cos(alpha) + x_currents(x,y,t)
    dot{x}_2 = u*u_max*sin(alpha) + y_currents(x,y,t)
    such that u in [0,1] and alpha in [0, 2pi]
    The controls are u and alpha.
    """

    def __init__(self,
                 u_max=[1.,1.],
                 d_max=0,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None,
                 x_current=None,
                 y_current=None):

        # set variables
        self.u_max = jnp.array(u_max)

        if self.u_max.size == 1:
            self.u_max = jnp.array([u_max,u_max])

        self.x_current = x_current
        self.y_current = y_current

        if control_space is None:
            control_space = sets.Box(lo=jnp.array([0, 0]),
                                     hi=jnp.array([1., 2 * jnp.pi]))
        if disturbance_space is None:
            disturbance_space = sets.Ball(center=jnp.zeros(2), radius=d_max)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE."""
        dx1 = self.u_max[0] * control[0] * jnp.cos(control[1]) + self.x_current(jnp.append(state, time)) + disturbance[0]
        dx2 = self.u_max[1] * control[0] * jnp.sin(control[1]) + self.y_current(jnp.append(state, time)) + disturbance[1]

        return jnp.array([dx1, dx2])

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
        alpha = jax.lax.atan2(self.u_max[1]*grad_value[1], self.u_max[0]*grad_value[0])
        # if min, go against the gradient direction
        if self.control_mode == 'min':
            alpha = alpha + jnp.pi
        return jnp.array([uOpt, alpha])

    def optimal_disturbance(self, state, time, grad_value):
        """Computes the optimal disturbance realized by the HJ PDE Hamiltonian."""
        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        return jnp.zeros_like(self.disturbance_space.extreme_point(disturbance_direction))

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        return (self.optimal_control(state, time, grad_value),
                self.optimal_disturbance(state, time, grad_value))


class Platform2DcurrentsDiscrete(Platform2Dcurrents):
    def optimal_control(self, state, time, grad_value):
        """Computes the optimal control realized by the HJ PDE Hamiltonian."""
        uOpt = jnp.array(1.0)
        # angle of px, py vector of gradient
        alpha = jax.lax.atan2(grad_value[1], grad_value[0])
        # if min, go against the gradient direction
        if self.control_mode == "min":
            alpha = alpha + jnp.pi
        # now make it discretized action space!
        disc_action = jnp.round(alpha / (jnp.pi / 4))
        continous_disc_action = disc_action * jnp.pi / 4
        return jnp.array([uOpt, continous_disc_action])
