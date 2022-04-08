from ocean_navigation_simulator.utils.analytical_fields.AnalyticalField import AnalyticalField
import jax.numpy as jnp
import numpy as np


class PeriodicDoubleGyre(AnalyticalField):
    """ The Periodic Double Gyre Analytical current Field.
    Note: the spatial domain is fixed to [0,2]x[0,1]
    Source:
    https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html#Sec7.1

        Attributes:
            spatial_domain
                Spatial Domain of the Gyres with currents e.g. [np.array([0, 0]), np.array([2, 1])]
            boundary_buffers:
                Margin to buffer the spatial domain with obstacles as boundary conditions e.g. [0.2, 0.2]
            spatial_output_shape
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

    def __init__(self, spatial_domain, boundary_buffers, spatial_output_shape, temporal_domain,
                 temporal_default_length, v_amplitude, epsilon_sep, period_time):
        # adjust spatial domain by boundary buffer
        spatial_domain[0] = spatial_domain[0] - np.array(boundary_buffers)
        spatial_domain[1] = spatial_domain[1] + np.array(boundary_buffers)

        super().__init__(spatial_domain=spatial_domain,
                         spatial_output_shape=spatial_output_shape,
                         temporal_domain=temporal_domain, temporal_default_length=temporal_default_length)

        self.v_amplitude = v_amplitude
        self.epsilon_sep = epsilon_sep
        self.period_time = period_time
        self.boundary_buffers = boundary_buffers  # in x and y direction

    def is_boundary(self, state, time):
        """Helper function to check if a state is in the obstacle."""
        del time
        x_boundary = jnp.logical_or(state[0] < self.spatial_domain.lo[0] + self.boundary_buffers[0],
                                    state[0] > self.spatial_domain.hi[0] - self.boundary_buffers[0])
        y_boundary = jnp.logical_or(state[1] < self.spatial_domain.lo[1] + self.boundary_buffers[1],
                                    state[1] > self.spatial_domain.hi[1] - self.boundary_buffers[1])

        return jnp.logical_or(x_boundary, y_boundary)

    def u_current_analytical(self, state, time):
        """Analytical Formula for u velocity of Periodic Double Gyre."""
        time = self.get_time_relative_to_t_0(time)
        w_angular_vel = 2*jnp.pi/self.period_time
        a = self.epsilon_sep*jnp.sin(w_angular_vel*time)
        b = 1 - 2*self.epsilon_sep*jnp.sin(w_angular_vel*time)
        f = a*jnp.power(a*state[0],2) + b*state[0]

        u_cur_out = -jnp.pi*self.v_amplitude*jnp.sin(jnp.pi*f)*jnp.cos(jnp.pi*state[1])
        return jnp.where(self.is_boundary(state, time), 0., u_cur_out)

    def v_current_analytical(self, state, time):
        """Analytical Formula for u velocity of Periodic Double Gyre."""
        time = self.get_time_relative_to_t_0(time)
        w_angular_vel = 2*jnp.pi/self.period_time
        a = self.epsilon_sep*jnp.sin(w_angular_vel*time)
        b = 1 - 2*self.epsilon_sep*jnp.sin(w_angular_vel*time)
        f = a*jnp.power(a*state[0], 2) + b*state[0]
        df_dx = 2*a*state[0] + b

        v_cur_out = jnp.pi*self.v_amplitude*jnp.cos(jnp.pi*f)*jnp.sin(jnp.pi*state[1])*df_dx
        return jnp.where(self.is_boundary(state, time), 0., v_cur_out)