from ocean_navigation_simulator.utils.analytical_fields.AnalyticalField import AnalyticalField
import jax.numpy as jnp


class FixedCurrentHighwayField(AnalyticalField):
    """ The Highway current Field.

        Attributes:
            spatial_output_shape:
                2D Tuple e.g. (100, 100) that represents the default output shape for get_subset
            temporal_domain:
                List e.g. [0, 100] capturing the temporal domain (enforced as limits in get_subset)
            temporal_default_length:
                integer e.g. 20 that represents default length of temporal dimension for get_subset
            y_range_highway:
                list representing the y-axis range of the highway current e.g. [3, 5]
            U_cur:
                strength of the current in space units/ time unit
        """

    def __init__(self, spatial_domain, spatial_output_shape, temporal_domain, temporal_default_length, y_range_highway, U_cur):
        super().__init__(spatial_domain, spatial_output_shape, temporal_domain, temporal_default_length)

        self.y_range_highway = y_range_highway
        self.U_cur = U_cur

    def u_current_analytical(self, state, time):
        u_cur_low = jnp.where(state[1] <= self.y_range_highway[1], self.U_cur, 0.)
        u_cur_out = jnp.where(self.y_range_highway[0] <= state[1] , u_cur_low, 0.)
        return u_cur_out

    def v_current_analytical(self, state, time):
        return 0.