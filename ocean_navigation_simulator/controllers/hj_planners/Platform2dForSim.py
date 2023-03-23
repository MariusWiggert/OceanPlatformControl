import functools
from typing import Union

import jax.lax
from hj_reachability.finite_differences import upwind_first
import jax.numpy as jnp
import xarray as xr
from jax import jit
from jax.lax import dynamic_slice, dynamic_slice_in_dim
from hj_reachability import dynamics, interpolation, sets
from hj_reachability.grid import Grid
from scipy.interpolate import interp1d


def transform_to_geographic_velocity(state, dx1, dx2):
    """Helper Function to transform dx1 and dx2 from m/s to the geographic_coordinate_system."""
    lon_delta_deg_per_s = 180 * dx1 / jnp.pi / 6371000 / jnp.cos(jnp.pi * state[1] / 180)
    lat_delta_deg_per_s = 180 * dx2 / jnp.pi / 6371000
    return jnp.array([lon_delta_deg_per_s, lat_delta_deg_per_s]).reshape(-1)


def geographic_transformation_matrix(state):
    """Helper Function to transform dx1 and dx2 from m/s to the geographic_coordinate_system.
    Output can be used as matrix multiplication d_x_geographic = M @ d_x_cartesian
    """
    x_transform = 180 / jnp.pi / 6371000 / jnp.cos(jnp.pi * state[1] / 180)
    y_transform = 180 / jnp.pi / 6371000

    return jnp.array([
        [x_transform, 0.],
        [0., y_transform],
    ])



class Platform2dForSim(dynamics.Dynamics):
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
    ):

        # set variables
        self.u_max = jnp.array(u_max)
        self.use_geographic_coordinate_system = use_geographic_coordinate_system

        # initialize the current interpolants with None, they are set in the planner method
        self.x_current, self.y_current = None, None

        # # obstacle operator (is overwritten if analytical_current with boundary obstacles)
        # self.obstacle_operator = lambda state, time, dx_out: dx_out

        control_space = sets.Box(lo=jnp.array([0, 0]), hi=jnp.array([1.0, 2 * jnp.pi]))

        disturbance_space = sets.Ball(center=jnp.zeros(2), radius=d_max)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def update_jax_interpolant(self, data_xarray: xr):
        """Creating an interpolant function from x,y,t grid and data
        Args:
            data_xarray: xarray containing variables water_u and water_v as matrices (T, Y, X)
        """

        # create 1D interpolation functions for running in the loop of the dynamics
        self.x_current = lambda state, time: interpolation.lin_interpo_1D(
            state,
            time,
            data_xarray["water_u"].fillna(0).data,
            data_xarray["lon"].data,
            data_xarray["lat"].data,
            data_xarray["relative_time"].data,
        )
        self.y_current = lambda state, time: interpolation.lin_interpo_1D(
            state,
            time,
            data_xarray["water_v"].fillna(0).data,
            data_xarray["lon"].data,
            data_xarray["lat"].data,
            data_xarray["relative_time"].data,
        )

    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE."""
        # dx is in m/s
        dx1 = (
            self.u_max * control[0] * jnp.cos(control[1])
            + self.x_current(state, time)
            + disturbance[0]
        )
        dx2 = (
            self.u_max * control[0] * jnp.sin(control[1])
            + self.y_current(state, time)
            + disturbance[1]
        )
        # now transfer it to deg lat/lon per second if use_geographic_coordinate_system
        dx_out = jnp.where(
            self.use_geographic_coordinate_system,
            transform_to_geographic_velocity(state, dx1, dx2),
            jnp.array([dx1, dx2]).reshape(-1),
        )
        return dx_out
        # return self.obstacle_operator(state, time, dx_out)

    @staticmethod
    def disturbance_jacobian(state, time):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])

    def optimal_control(self, state, time, grad_value):
        """Computes the optimal control realized by the HJ PDE Hamiltonian."""
        uOpt = jnp.array(1.0)
        # angle of px, py vector of gradient
        alpha = jax.lax.atan2(grad_value[1], grad_value[0])
        # if min, go against the gradient direction
        if self.control_mode == "min":
            alpha = alpha + jnp.pi
        return jnp.array([uOpt, alpha])

    def optimal_disturbance(self, state, time, grad_value):
        """Computes the optimal disturbance realized by the HJ PDE Hamiltonian."""
        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        return self.disturbance_space.extreme_point(disturbance_direction)

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        return (
            self.optimal_control(state, time, grad_value),
            self.optimal_disturbance(state, time, grad_value),
        )

    ### Overwriting lower functions for fast closed-loop control
    def get_opt_ctrl_from_values(self, x, grid, time, times, all_values,
                                 upwind_scheme=upwind_first.WENO3, n_temporal_buffer_idx=6, n_spatial_buffer_idx=10):
        """Function that computes the optimal ctrl and distr for a point state at time based on the value function.

        Input Params:
        - x                 state at which to get optimal ctrl and distr
        - grid              same grid as used to solve the PDE
        - time              time of the value function
        - times             times vector used for solving
        - all_values        values from the PDE solve (T, ...)
        - upwind_scheme     (Optional) schema to calculate the value function gradient. WENO3 is used as default.
        - n_temporal_buffer_idx (Optional) number of temporal buffer indices to use for the time sub-setting
        - n_spatial_buffer_idx (Optional) number of spatial buffer indices to use for the gradient calculation

        Output:
        - u_opt              optimal control at x at time (nCtrl, )
        - d_opt              optimal disturbance at x at time (nDistr, )
        """
        # Step 0: subset the grid around the current state
        subsetted_grid, subsetted_reach_times, subsetted_values = self.create_subsets(
            time, times, all_values, grid, x, n_temporal_buffer_idx, n_spatial_buffer_idx)

        # Step 1: interpolate the value function for the specific time along the time axis
        val_at_t = interp1d(subsetted_reach_times, subsetted_values, axis=0, kind='linear')(time).squeeze()

        # Step 2: get_opt_ctrl_from_values_jit
        u_opt, d_opt = self.get_opt_ctrl_from_val_at_t_jit(state=x, time=time, grid=subsetted_grid,
                                                           val_at_t=val_at_t, upwind_scheme=upwind_scheme)
        return u_opt, d_opt

    @staticmethod
    @functools.partial(jit, static_argnames=("n_temporal_buffer_idx", "n_spatial_buffer_idx"))
    def create_subsets(time, times, all_values, grid, state, n_temporal_buffer_idx, n_spatial_buffer_idx):
        """Helper function to create a grid object that is a subset of the original grid around the current state and time."""
        # Step 1: Get nearest index in the spatial grid
        nearest_index = grid.nearest_index(state=state)
        rel_time_index = jnp.argmin(jnp.abs(times - time))
        # pre-step, get the starting indices for all dimensions
        t_idx_start = jnp.maximum(rel_time_index - n_temporal_buffer_idx, 0)
        x_idx_start = jnp.maximum(nearest_index[0] - n_spatial_buffer_idx, 0)
        y_idx_start = jnp.maximum(nearest_index[1] - n_spatial_buffer_idx, 0)
        # Subset the reach times
        subsetted_reach_times = dynamic_slice_in_dim(times, t_idx_start, 2 * n_temporal_buffer_idx)
        # Step 2: create a smaller grid as subset of planner.grid around the nearest index
        subsetted_states = dynamic_slice(grid.states, (x_idx_start, y_idx_start, 0),
                                         (2 * n_spatial_buffer_idx, 2 * n_spatial_buffer_idx, grid.ndim))
        # subset the value function in planner.all_values the same way (Time, X, Y)
        subsetted_values = dynamic_slice(all_values, (t_idx_start, x_idx_start, y_idx_start),
                                         (2 * n_temporal_buffer_idx, 2 * n_spatial_buffer_idx, 2 * n_spatial_buffer_idx))
        # subsetted_values = dynamic_slice(all_values, (t_idx_start, 0, 0),
        #                                  (2 * n_temporal_buffer_idx, all_values.shape[1], all_values.shape[2]))

        new_coords = *(dynamic_slice_in_dim(cord_vec, idx_start, 2 * n_spatial_buffer_idx) for cord_vec, idx_start in
                       zip(grid.coordinate_vectors, [x_idx_start, y_idx_start])),
        boundaries = jnp.take(jnp.vstack(new_coords), jnp.array([0, -1]), axis=1)
        subsetted_domain = sets.Box(lo=boundaries[:, 0], hi=boundaries[:, 1])
        # create subsetted grid object
        subsetted_grid = Grid(
            subsetted_states,
            subsetted_domain,
            new_coords,
            grid.spacings,
            grid.boundary_conditions
        )
        return subsetted_grid, subsetted_reach_times, subsetted_values

    @functools.partial(jax.jit, static_argnames=("self", "upwind_scheme"))
    def get_opt_ctrl_from_val_at_t_jit(self, state, time, grid, val_at_t, upwind_scheme=upwind_first.WENO3):
        """Function that computes the optimal ctrl and distr for a state at time based on val_at_t.
        Input Params:
        - state             state at which to get optimal ctrl and distr
        - time              time of the value function
        - grid              grid object over which val_at_t is defined
        - val_at_t          values from the PDE solve at time over grid
        - upwind_scheme     (Optional) schema to calculate the value function gradient. WENO3 is used as default.

        Output:
        - u_opt              optimal control at x at time (nCtrl, )
        - d_opt              optimal disturbance at x at time (nDistr, )
        """

        # Step 1: get center approximation of gradient at current point x
        grad_at_x_cur = grid.interpolate(values=grid.grad_values(values=val_at_t, upwind_scheme=upwind_scheme), state=state.flatten())

        # Step 2: get u_opt and d_opt
        u_opt, d_opt = self.optimal_control_and_disturbance(state=state.flatten(), time=time, grad_value=grad_at_x_cur)

        return u_opt, d_opt



class Platform2dForSimAffine(dynamics.ControlAndDisturbanceAffineDynamics):
    """The 2D Ocean going Platform class on a dynamic current field.
    This class is for use with the ocean_platform simulator

    Dynamics:
    dot{x}_1 = u_x + x_currents(x,y,t)
    dot{x}_2 = u_y + y_currents(x,y,t)
    such that ||(u_x, u_y)|| < u_max
    The controls are u_x and u_y.

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
    ):

        # set variables
        self.use_geographic_coordinate_system = use_geographic_coordinate_system

        # initialize the current interpolants with None, they are set in the planner method
        self.x_current, self.y_current = None, None

        # # obstacle operator (is overwritten if analytical_current with boundary obstacles)
        # self.obstacle_operator = lambda state, time, dx_out: dx_out
        self.u_max = u_max
        control_space = sets.Ball(center=jnp.zeros(2), radius=u_max)

        disturbance_space = sets.Ball(center=jnp.zeros(2), radius=d_max)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def __call__(self, state, control, disturbance, time):
        """Implements the affine dynamics `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`.
        This just adds the transformation to the geographical coordinate system.
        """
        cartesian_dynamics = (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                              self.disturbance_jacobian(state, time) @ disturbance)

        return jnp.where(
            self.use_geographic_coordinate_system,
            geographic_transformation_matrix(state) @ cartesian_dynamics,
            cartesian_dynamics)

    def update_jax_interpolant(self, data_xarray: xr):
        """Creating an interpolant function from x,y,t grid and data
        Args:
            data_xarray: xarray containing variables water_u and water_v as matrices (T, Y, X)
        """

        # create 1D interpolation functions for running in the loop of the dynamics
        self.x_current = lambda state, time: interpolation.lin_interpo_1D(
            state,
            time,
            data_xarray["water_u"].fillna(0).data,
            data_xarray["lon"].data,
            data_xarray["lat"].data,
            data_xarray["relative_time"].data,
        )
        self.y_current = lambda state, time: interpolation.lin_interpo_1D(
            state,
            time,
            data_xarray["water_v"].fillna(0).data,
            data_xarray["lon"].data,
            data_xarray["lat"].data,
            data_xarray["relative_time"].data,
        )

    def open_loop_dynamics(self, state, time):
        """Implements the open-loop dynamics (without controls)."""
        dx1 = self.x_current(state, time)
        dx2 = self.y_current(state, time)
        return jnp.array([dx1, dx2])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[1.0, 0.0],
                          [0.0, 1.0]])

    def control_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
        ])

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        # Note: might be wrong and we have to do it on individual elements, trying as this would be cleaner.
        cartesian_partial_max_magnitudes = super().partial_max_magnitudes(state, time, value, grad_value_box)
        return jnp.where(
            self.use_geographic_coordinate_system,
            jnp.abs(geographic_transformation_matrix(state) @ cartesian_partial_max_magnitudes),
            cartesian_partial_max_magnitudes)