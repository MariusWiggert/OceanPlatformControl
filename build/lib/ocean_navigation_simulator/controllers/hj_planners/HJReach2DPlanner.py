import os
import pickle

import numpy as np
import jax.numpy as jnp
import warnings
import math

from ocean_navigation_simulator.controllers.hj_planners.Platform2dForSim import Platform2dForSim
from ocean_navigation_simulator.controllers.hj_planners.HJPlannerBase import HJPlannerBase
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatioTemporalPoint, SpatialPoint
from ocean_navigation_simulator.environment.Platform import PlatformAction
import hj_reachability as hj
import xarray as xr
from typing import Union, Optional


class HJReach2DPlanner(HJPlannerBase):
    """ Reachability planner for 2D (lat, lon) reachability computation."""
    gpus: float = 1.0

    def get_x_from_full_state(self, x: Union[PlatformState, SpatioTemporalPoint, SpatialPoint]) -> jnp.ndarray:
        return jnp.array(x.__array__())[:2]

    def get_dim_dynamical_system(self) -> hj.dynamics.Dynamics:
        """Initialize 2D (lat, lon) Platform dynamics in deg/s."""
        return Platform2dForSim(
            u_max=self.specific_settings['platform_dict']['u_max_in_mps'], d_max=self.specific_settings['d_max'],
            use_geographic_coordinate_system= self.specific_settings['use_geographic_coordinate_system'],
            control_mode='min', disturbance_mode='max')

    def initialize_hj_grid(self, xarray: xr) -> None:
        """Initialize the dimensional grid in degrees lat, lon"""
        # initialize grid using the grids_dict x-y shape as shape
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj.sets.Box(
                lo=np.array([xarray['lon'][0].item(), xarray['lat'][0].item()]),
                hi=np.array([xarray['lon'][-1].item(), xarray['lat'][-1].item()])
            ),
            shape=(xarray['lon'].size, xarray['lat'].size)
        )

    def get_initial_values(self, direction) -> jnp.ndarray:
        """ Setting the initial values for the HJ PDE solver."""
        if direction == "forward":
            center = self.x_t
            return hj.shapes.shape_ellipse(
                grid=self.nonDimGrid,
                center=self._get_non_dim_state(self.get_x_from_full_state(center)),
                radii=self.specific_settings['initial_set_radii']/self.characteristic_vec
            )
        elif direction == "backward":
            center = self.problem.end_region
            return hj.shapes.shape_ellipse(
                grid=self.nonDimGrid,
                center=self._get_non_dim_state(self.get_x_from_full_state(center)),
                radii=[self.problem.target_radius,self.problem.target_radius]/self.characteristic_vec
            )
        elif direction == "multi-time-reach-back":
            center = self.problem.end_region
            signed_distance = hj.shapes.shape_ellipse(
                grid=self.nonDimGrid,
                center=self._get_non_dim_state(self.get_x_from_full_state(center)),
                radii=[self.problem.target_radius, self.problem.target_radius] / self.characteristic_vec
            )
            return np.maximum(signed_distance, np.zeros(signed_distance.shape))
        else:
            raise ValueError("Direction in specific_settings of HJPlanner needs to be forward, backward, or multi-reach-back.")

    def save_planner_state(self, folder):
        os.makedirs(folder, exist_ok = True)
        # Settings
        with open(folder + 'specific_settings.pickle', 'wb') as file:
            pickle.dump(self.specific_settings, file)
        # Used in Replanning
        with open(folder + 'last_fmrc_idx_planned_with.pickle', 'wb') as file:
            pickle.dump(self.last_fmrc_idx_planned_with, file)
        # Used in Interpolation
        with open(folder + 'all_values.pickle', 'wb') as file:
            pickle.dump(self.all_values, file)
        with open(folder + 'reach_times.pickle', 'wb') as file:
            pickle.dump(self.reach_times, file)
        with open(folder + 'grid.pickle', 'wb') as file:
            pickle.dump(self.grid, file)
        with open(folder + 'current_data_t_0.pickle', 'wb') as file:
            pickle.dump(self.current_data_t_0, file)
        with open(folder + 'current_data_t_T.pickle', 'wb') as file:
            pickle.dump(self.current_data_t_T, file)
        # Used in Start Sampling
        with open(folder + 'characteristic_vec.pickle', 'wb') as file:
            pickle.dump(self.characteristic_vec, file)
        with open(folder + 'offset_vec.pickle', 'wb') as file:
            pickle.dump(self.characteristic_vec, file)
        with open(folder + 'initial_values.pickle', 'wb') as file:
            pickle.dump(self.initial_values, file)

    @staticmethod
    def from_saved_planner_state(folder, problem: NavigationProblem, verbose: Optional[int] = 0):
        # Settings
        with open(folder + 'specific_settings.pickle', 'rb') as file:
            specific_settings= pickle.load(file)

        planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)

        # Used in Replanning
        with open(folder + 'last_fmrc_idx_planned_with.pickle', 'rb') as file:
            planner.last_fmrc_idx_planned_with = pickle.load(file)
        # Used in Interpolation
        with open(folder + 'all_values.pickle', 'rb') as file:
            planner.all_values = pickle.load(file)
        with open(folder + 'reach_times.pickle', 'rb') as file:
            planner.reach_times = pickle.load(file)
        with open(folder + 'grid.pickle', 'rb') as file:
            planner.grid = pickle.load(file)
        with open(folder + 'current_data_t_0.pickle', 'rb') as file:
            planner.current_data_t_0 = pickle.load(file)
        with open(folder + 'current_data_t_T.pickle', 'rb') as file:
            planner.current_data_t_T = pickle.load(file)
        # Used in Start Sampling
        with open(folder + 'characteristic_vec.pickle', 'rb') as file:
            planner.characteristic_vec = pickle.load(file)
        with open(folder + 'offset_vec.pickle', 'rb') as file:
            planner.offset_vec = pickle.load(file)
        with open(folder + 'initial_values.pickle', 'rb') as file:
            planner.initial_values = pickle.load(file)
        planner.set_interpolator()

        return planner


class HJReach2DPlannerWithErrorHeuristic(HJReach2DPlanner):
    #TODO: this does not work after redesign with the state and action classes, needs to be adjusted if used.
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
        lon_target, lat_target = self.problem.end_region.lon.deg, self.problem.end_region.lat.deg

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