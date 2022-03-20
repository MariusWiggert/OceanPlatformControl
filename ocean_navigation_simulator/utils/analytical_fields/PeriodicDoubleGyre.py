from ocean_navigation_simulator.utils.analytical_fields.AnalyticalField import AnalyticalField
import jax.numpy as jnp
import numpy as np


class PeriodicDoubleGyre(AnalyticalField):
    """ The Periodic Double Gyre Analytical current Field.
    Note: the spatial domain is fixed to [0,2]x[0,1]
    Source:
    https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html#Sec7.1

        Attributes:
            spatial_output_shape:
                2D Tuple e.g. (100, 100) that represents the default output shape for get_subset
            temporal_domain:
                List e.g. [0, 100] capturing the temporal domain (enforced as limits in get_subset)
            temporal_default_length:
                integer e.g. 20 that represents default length of temporal dimension for get_subset
            v_amplitude:
                float representing maximum current strength in space units/time units.
            epsilon_sep:
                float >= 0 representing the magnitude of oscillation of the gyre around x=1.
                The flow becomes time-independent at epsilon_sep = 0
            period_time:
                positive float of a full period time of an oscillation in time units.
        """

    def __init__(self, spatial_output_shape, temporal_domain, temporal_default_length, v_amplitude, epsilon_sep, period_time):

        super().__init__(spatial_domain = [np.array([0, 0]), np.array([2, 1])],
                         spatial_output_shape =spatial_output_shape,
                         temporal_domain=temporal_domain, temporal_default_length=temporal_default_length)

        self.v_amplitude = v_amplitude
        self.epsilon_sep = epsilon_sep
        self.period_time = period_time

    def u_current_analytical(self, state, time):
        """Analytical Formula for u velocity of Periodic Double Gyre."""
        w_angular_vel = 2*jnp.pi/self.period_time
        a = self.epsilon_sep*jnp.sin(w_angular_vel*time)
        b = 1 - 2*self.epsilon_sep*jnp.sin(w_angular_vel*time)
        f = a*jnp.power(a*state[0],2) + b*state[0]

        u_cur_out = -jnp.pi*self.v_amplitude*jnp.sin(jnp.pi*f)*jnp.cos(jnp.pi*state[1])
        return u_cur_out

    def v_current_analytical(self, state, time):
        """Analytical Formula for u velocity of Periodic Double Gyre."""
        w_angular_vel = 2*jnp.pi/self.period_time
        a = self.epsilon_sep*jnp.sin(w_angular_vel*time)
        b = 1 - 2*self.epsilon_sep*jnp.sin(w_angular_vel*time)
        f = a*jnp.power(a*state[0],2) + b*state[0]
        df_dx = 2*a*state[0] + b

        v_cur_out = jnp.pi*self.v_amplitude*jnp.cos(jnp.pi*f)*jnp.sin(jnp.pi*state[1])*df_dx
        return v_cur_out