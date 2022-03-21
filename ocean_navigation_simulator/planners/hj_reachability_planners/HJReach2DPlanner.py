from ocean_navigation_simulator.planners.hj_reachability_planners.HJPlannerBase import HJPlannerBase
import numpy as np
import jax.numpy as jnp
import warnings
import math
from ocean_navigation_simulator.planners.hj_reachability_planners.platform_2D_for_sim import Platform2D_for_sim, Platform2D_for_sim_with_disturbance
import hj_reachability as hj


class HJReach2DPlanner(HJPlannerBase):
    """ Reachability planner for 2D (lat, lon) reachability computation."""

    def get_x_from_full_state(self, x):
        return x[:2]

    def get_dim_dynamical_system(self):
        """Initialize 2D (lat, lon) Platform dynamics in deg/s."""
        # space coefficient is fixed for now as we run in deg/s (same as the simulator)
        space_coeff = 1. / self.conv_m_to_deg
        if 'd_max' in self.specific_settings and self.specific_settings['direction'] != "multi-reach-back":
            print("High-Level Planner: Running with d_max")
            return Platform2D_for_sim_with_disturbance(u_max=self.dyn_dict['u_max'], d_max=self.specific_settings['d_max'],
                                      space_coeff=space_coeff, control_mode='min', disturbance_mode='max')
        else:
            return Platform2D_for_sim(u_max=self.dyn_dict['u_max'], space_coeff=space_coeff, control_mode='min', disturbance_mode='max')


    def initialize_hj_grid(self, grids_dict):
        """Initialize the dimensional grid in degrees lat, lon"""
        # initialize grid using the grids_dict x-y shape as shape
        self.grid = hj.Grid.from_grid_definition_and_initial_values(
            domain=hj.sets.Box(
                lo=np.array([grids_dict['x_grid'][0], grids_dict['y_grid'][0]]),
                hi=np.array([grids_dict['x_grid'][-1], grids_dict['y_grid'][-1]])),
            shape=(len(grids_dict['x_grid']), len(grids_dict['y_grid'])))

    def get_initial_values(self, direction):
        if direction == "forward":
            center = self.x_t
            return hj.shapes.shape_ellipse(grid=self.nonDimGrid,
                                           center=self.get_non_dim_state(self.get_x_from_full_state(center.flatten())),
                                           radii=self.specific_settings['initial_set_radii']/self.characteristic_vec)
        elif direction == "backward":
            center = self.x_T
            return hj.shapes.shape_ellipse(grid=self.nonDimGrid,
                                           center=self.get_non_dim_state(self.get_x_from_full_state(center.flatten())),
                                           radii=[self.problem.x_T_radius, self.problem.x_T_radius] / self.characteristic_vec)
        elif direction == "multi-reach-back":
            center = self.x_T
            signed_distance = hj.shapes.shape_ellipse(grid=self.nonDimGrid,
                                           center=self.get_non_dim_state(self.get_x_from_full_state(center.flatten())),
                                           radii=[self.problem.x_T_radius, self.problem.x_T_radius] / self.characteristic_vec)
            return np.maximum(signed_distance, np.zeros(signed_distance.shape))
        else:
            raise ValueError("Direction in specific_settings of HJPlanner needs to be forward, backward, or multi-reach-back.")


class HJReach2DPlannerWithErrorHeuristic(HJReach2DPlanner):
    """Version of the HJReach2DPlanner that contains a heuristic to adjust the control, when the locally sensed
    current error (forecasted_vec - sensed_vec) is above a certain threshold.
    """
    def __init__(self, problem, specific_settings, conv_m_to_deg):
        # initialize Planner superclass
        super().__init__(problem, specific_settings, conv_m_to_deg)
        # check if EVM_threshold is set
        if not 'EVM_threshold' in self.specific_settings:
            raise ValueError("EVM_threshold is not set, needs to be in specific_settings.")

    def get_next_action(self, state, trajectory):
        """Adjust the angle based on the Error Vector Magnitude.
        EVM = ||forecasted_vec_{t-1} + sensed_vec_{t-1}||_2
        """

        # Step 0: get the optimal control from the classic approach
        if self.specific_settings['direction'] == 'forward':
            u_out = super().get_u_from_vectors(state, ctrl_vec='dir')
        else:
            # check if time is outside times and through warning if yes but continue.
            rel_time = state[3] - self.current_data_t_0
            if rel_time > self.reach_times[-1]:
                warnings.warn("Extrapolating time beyond the reach_times, should replan.", RuntimeWarning)
                rel_time = self.reach_times[-1]
            u_out, _ = self.nondim_dynamics.dimensional_dynamics.get_opt_ctrl_from_values(
                grid=self.grid, x=self.get_x_from_full_state(state),
                time=rel_time,
                times=self.reach_times, all_values=self.all_values)

        # default u_out if error is below threshold
        u_out = np.asarray(u_out.reshape(-1, 1))
        # because first step we can't sense
        if trajectory.shape[1] == 1:
            return u_out

        # Step 1: check if EVM of forecast in last time step is above threshold
        # This is in deg/s
        ds = trajectory[:2, -1] - trajectory[:2, -2]
        dt = trajectory[3, -1] - trajectory[3, -2]
        last_sensed_vec = (ds / dt)
        # correct to rel_time for querying the forecasted current
        rel_time = state[3] - self.current_data_t_0
        # This is in deg/s
        cur_forecasted = self.nondim_dynamics.dimensional_dynamics(state[:2], jnp.array([0, 0]), jnp.array([0, 0]), rel_time)
        u_straight = self.get_straight_line_action(state)
        # compute EVM
        EVM = jnp.linalg.norm(cur_forecasted - last_sensed_vec)/self.nondim_dynamics.dimensional_dynamics.space_coeff
        # check if above threshold, if yes do weighting heuristic
        if EVM >= self.specific_settings['EVM_threshold']:
            print("EVM above threshold = ", EVM)
            basis = EVM + self.specific_settings['EVM_threshold']
            w_straight_line = EVM / basis
            w_fmrc_planned = self.specific_settings['EVM_threshold'] / basis
            print("angle_before: ", u_out[1])
            print("angle_straight: ", u_straight[1])
            angle_weighted = np.array(w_fmrc_planned * u_out[1] + w_straight_line * u_straight[1])[0]
            u_out = np.asarray([1, angle_weighted]).reshape(-1, 1)
            print("new_angle: ", u_out[1])

        return u_out

    def get_straight_line_action(self, x_t):
        """Go in the direction of the target with full power. See superclass for args and return value."""

        lon, lat = x_t[0][0], x_t[1][0]
        lon_target, lat_target = self.x_T[0], self.x_T[1]

        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # go there full power
        u_dir = np.array([[dlon / mag], [dlat / mag]])
        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        # make sure the angle is positive
        if u_out[1] < 0:
            u_out[1] = u_out[1] + 2*np.pi
        return u_out




