import os
import pickle

import numpy as np
import jax.numpy as jnp
import warnings
import math
import scipy

from ocean_navigation_simulator.controllers.hj_planners.Platform2dSeaweedForSim import (
    Platform2dSeaweedForSim,
)
from ocean_navigation_simulator.controllers.hj_planners.HJPlannerBase import HJPlannerBase
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatioTemporalPoint,
    SpatialPoint,
)
import hj_reachability as hj
import xarray as xr
from typing import Union, Optional, Dict


class HJBSeaweed2DPlanner(HJPlannerBase):
    """Reachability planner for 2D (lat, lon) reachability computation."""

    gpus: float = 1.0

    def __init__(
        self,
        arena: ArenaFactory,
        problem: NavigationProblem,
        specific_settings: Optional[Dict] = ...,
    ):

        # get arena object for accessing seaweed growth model
        self.arena = arena

        super().__init__(problem, specific_settings)

    def get_x_from_full_state(
        self, x: Union[PlatformState, SpatioTemporalPoint, SpatialPoint]
    ) -> jnp.ndarray:
        return jnp.array(x.__array__())[:2]

    def get_dim_dynamical_system(self) -> hj.dynamics.Dynamics:
        """Initialize 2D (lat, lon) Platform dynamics in deg/s."""
        return Platform2dSeaweedForSim(
            u_max=self.specific_settings["platform_dict"]["u_max_in_mps"],
            d_max=self.specific_settings["d_max"],
            use_geographic_coordinate_system=self.specific_settings[
                "use_geographic_coordinate_system"
            ],
            control_mode="min",
            disturbance_mode="max",
        )

    def initialize_hj_grid(self, xarray: xr) -> None:
        """Initialize the dimensional grid in degrees lat, lon"""
        # initialize grid using the grids_dict x-y shape as shape
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj.sets.Box(
                lo=np.array([xarray["lon"][0].item(), xarray["lat"][0].item()]),
                hi=np.array([xarray["lon"][-1].item(), xarray["lat"][-1].item()]),
            ),
            shape=(xarray["lon"].size, xarray["lat"].size),
        )

    def get_initial_values(self, direction) -> jnp.ndarray:
        """Setting the initial values for the HJ PDE solver."""
        if direction == "forward":
            center = self.x_t
            return hj.shapes.shape_ellipse(
                grid=self.nonDimGrid,
                center=self._get_non_dim_state(self.get_x_from_full_state(center)),
                radii=self.specific_settings["initial_set_radii"] / self.characteristic_vec,
            )
        elif direction == "backward":
            center = self.problem.end_region
            return hj.shapes.shape_ellipse(
                grid=self.nonDimGrid,
                center=self._get_non_dim_state(self.get_x_from_full_state(center)),
                radii=[self.problem.target_radius, self.problem.target_radius]
                / self.characteristic_vec,
            )
        elif direction == "multi-time-reach-back":
            center = self.problem.end_region
            signed_distance = hj.shapes.shape_ellipse(
                grid=self.nonDimGrid,
                center=self._get_non_dim_state(self.get_x_from_full_state(center)),
                radii=[self.problem.target_radius, self.problem.target_radius]
                / self.characteristic_vec,
            )
            return np.maximum(signed_distance, np.zeros(signed_distance.shape))
        else:
            raise ValueError(
                "Direction in specific_settings of HJPlanner needs to be forward, backward, or multi-reach-back."
            )

    # Functions to access the Value Function from outside #
    def set_interpolator(self):
        """Helper Function to create an interpolator for the value function for fast computation."""
        ttr_values = self.all_values - self.all_values.min(axis=(1, 2))[:, None, None]
        ttr_values = (
            ttr_values
            / np.abs(self.all_values.min())
            * (self.reach_times[-1] - self.reach_times[0])
            / 3600
        )

        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            points=(
                self.current_data_t_0 + self.reach_times,
                self.grid.states[:, 0, 0],
                self.grid.states[0, :, 1],
            ),
            values=ttr_values,
            method="linear",
        )

    def termination_condn(self, x, t):
        """Helper function to determine if target region is reached"""
        x_target = jnp.array(self.problem.end_region.__array__())
        r = self.problem.target_radius
        return np.linalg.norm(x_target - x) <= r

    def save_planner_state(self, folder):
        os.makedirs(folder, exist_ok=True)
        # Settings
        with open(folder + "specific_settings.pickle", "wb") as file:
            pickle.dump(self.specific_settings, file)
        # Used in Replanning
        with open(folder + "last_fmrc_idx_planned_with.pickle", "wb") as file:
            pickle.dump(self.last_fmrc_idx_planned_with, file)
        # Used in Interpolation
        with open(folder + "all_values.pickle", "wb") as file:
            pickle.dump(self.all_values, file)
        with open(folder + "reach_times.pickle", "wb") as file:
            pickle.dump(self.reach_times, file)
        with open(folder + "grid.pickle", "wb") as file:
            pickle.dump(self.grid, file)
        with open(folder + "current_data_t_0.pickle", "wb") as file:
            pickle.dump(self.current_data_t_0, file)
        with open(folder + "current_data_t_T.pickle", "wb") as file:
            pickle.dump(self.current_data_t_T, file)
        # Used in Start Sampling
        with open(folder + "characteristic_vec.pickle", "wb") as file:
            pickle.dump(self.characteristic_vec, file)
        with open(folder + "offset_vec.pickle", "wb") as file:
            pickle.dump(self.characteristic_vec, file)
        with open(folder + "initial_values.pickle", "wb") as file:
            pickle.dump(self.initial_values, file)

    @staticmethod
    def from_saved_planner_state(folder, problem: NavigationProblem, verbose: Optional[int] = 0):
        # Settings
        with open(folder + "specific_settings.pickle", "rb") as file:
            specific_settings = pickle.load(file)

        planner = HJBSeaweed2DPlanner(problem=problem, specific_settings=specific_settings)

        # Used in Replanning
        with open(folder + "last_fmrc_idx_planned_with.pickle", "rb") as file:
            planner.last_fmrc_idx_planned_with = pickle.load(file)
        # Used in Interpolation
        with open(folder + "all_values.pickle", "rb") as file:
            planner.all_values = pickle.load(file)
        with open(folder + "reach_times.pickle", "rb") as file:
            planner.reach_times = pickle.load(file)
        with open(folder + "grid.pickle", "rb") as file:
            planner.grid = pickle.load(file)
        with open(folder + "current_data_t_0.pickle", "rb") as file:
            planner.current_data_t_0 = pickle.load(file)
        with open(folder + "current_data_t_T.pickle", "rb") as file:
            planner.current_data_t_T = pickle.load(file)
        # Used in Start Sampling
        with open(folder + "characteristic_vec.pickle", "rb") as file:
            planner.characteristic_vec = pickle.load(file)
        with open(folder + "offset_vec.pickle", "rb") as file:
            planner.offset_vec = pickle.load(file)
        with open(folder + "initial_values.pickle", "rb") as file:
            planner.initial_values = pickle.load(file)
        planner.set_interpolator()

        return planner


class HJReach2DSeaweedPlannerWithErrorHeuristic(HJBSeaweed2DPlanner):
    # TODO: this does not work after redesign with the state and action classes, needs to be adjusted if used.
    """Version of the HJReach2DPlanner that contains a heuristic to adjust the control, when the locally sensed
    current error (forecasted_vec - sensed_vec) is above a certain threshold.
    """

    def __init__(self, problem, specific_settings, conv_m_to_deg):
        # initialize Planner superclass
        super().__init__(problem, specific_settings, conv_m_to_deg)
        # check if EVM_threshold is set
        if "EVM_threshold" not in self.specific_settings:
            raise ValueError("EVM_threshold is not set, needs to be in specific_settings.")

    def get_next_action(self, state, trajectory):
        """Adjust the angle based on the Error Vector Magnitude.
        EVM = ||forecasted_vec_{t-1} + sensed_vec_{t-1}||_2
        """

        # Step 0: get the optimal control from the classic approach
        if self.specific_settings["direction"] == "forward":
            u_out = super().get_u_from_vectors(state, ctrl_vec="dir")
        else:
            # check if time is outside times and through warning if yes but continue.
            rel_time = state[3] - self.current_data_t_0
            if rel_time > self.reach_times[-1]:
                warnings.warn(
                    "Extrapolating time beyond the reach_times, should replan.", RuntimeWarning
                )
                rel_time = self.reach_times[-1]
            u_out, _ = self.nondim_dynamics.dimensional_dynamics.get_opt_ctrl_from_values(
                grid=self.grid,
                x=self.get_x_from_full_state(state),
                time=rel_time,
                times=self.reach_times,
                all_values=self.all_values,
            )

        # default u_out if error is below threshold
        u_out = np.asarray(u_out.reshape(-1, 1))
        # because first step we can't sense
        if trajectory.shape[1] == 1:
            return u_out

        # Step 1: check if EVM of forecast in last time step is above threshold
        # This is in deg/s
        ds = trajectory[:2, -1] - trajectory[:2, -2]
        dt = trajectory[3, -1] - trajectory[3, -2]
        last_sensed_vec = ds / dt
        # correct to rel_time for querying the forecasted current
        rel_time = state[3] - self.current_data_t_0
        # This is in deg/s
        cur_forecasted = self.nondim_dynamics.dimensional_dynamics(
            state[:2], jnp.array([0, 0]), jnp.array([0, 0]), rel_time
        )
        u_straight = self.get_straight_line_action(state)
        # compute EVM
        EVM = (
            jnp.linalg.norm(cur_forecasted - last_sensed_vec)
            / self.nondim_dynamics.dimensional_dynamics.space_coeff
        )
        # check if above threshold, if yes do weighting heuristic
        if EVM >= self.specific_settings["EVM_threshold"]:
            print("EVM above threshold = ", EVM)
            basis = EVM + self.specific_settings["EVM_threshold"]
            w_straight_line = EVM / basis
            w_fmrc_planned = self.specific_settings["EVM_threshold"] / basis
            print("angle_before: ", u_out[1])
            print("angle_straight: ", u_straight[1])
            angle_weighted = np.array(w_fmrc_planned * u_out[1] + w_straight_line * u_straight[1])[
                0
            ]
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
            u_out[1] = u_out[1] + 2 * np.pi
        return u_out


    def _plan(self, x_t: PlatformState):
        """Main function where the reachable front is computed.
        Args:
            x_t: Platform state used as start/target of HJ Reachability computation, depending on 'direction'.
        """

        # run data checks if the right current data is loaded in the interpolation function
        self._check_data_settings_compatibility(x_t=x_t)

        # Step 2: depending on the reachability direction run the respective algorithm
        if self.specific_settings["direction"] == "forward":
            self._run_hj_reachability(
                initial_values=self.get_initial_values(direction="forward"),
                t_start=x_t.date_time,
                T_max_in_seconds=self.specific_settings["T_goal_in_seconds"],
                dir="forward",
                x_reach_end=None,  # self.get_x_from_full_state(self.problem.end_region)
            )
            self._extract_trajectory(x_start=self.get_x_from_full_state(self.problem.end_region))

        elif self.specific_settings["direction"] == "backward":
            # Note: no trajectory is extracted as the value function is used for closed-loop control
            self._run_hj_reachability(
                initial_values=self.get_initial_values(direction="backward"),
                t_start=x_t.date_time,
                T_max_in_seconds=self.specific_settings["T_goal_in_seconds"],
                dir="backward",
            )
            self._extract_trajectory(x_start=self.get_x_from_full_state(x_t))
            # arrange to forward times by convention for plotting and open-loop control
            self._flip_value_func_to_forward_times()

        elif self.specific_settings["direction"] == "forward-backward":
            # Step 1: run the set forward to get the earliest possible arrival time
            self._run_hj_reachability(
                initial_values=self.get_initial_values(direction="forward"),
                t_start=x_t.date_time,
                T_max_in_seconds=self.specific_settings["T_goal_in_seconds"],
                dir="forward",
                x_reach_end=self.get_x_from_full_state(self.problem.end_region),
            )
            # Step 2: run the set backwards from the earliest arrival time backwards
            _, t_earliest_in_seconds = self._get_t_earliest_for_target_region()
            print("earliest for target region is ", t_earliest_in_seconds)
            self._run_hj_reachability(
                initial_values=self.get_initial_values(direction="backward"),
                t_start=x_t.date_time,
                T_max_in_seconds=t_earliest_in_seconds
                + self.specific_settings["fwd_back_buffer_in_seconds"],
                dir="backward",
            )
            self._extract_trajectory(x_start=self.get_x_from_full_state(x_t))
            # arrange to forward times by convention for plotting and open-loop control
            self._flip_value_func_to_forward_times()
        elif self.specific_settings["direction"] == "multi-time-reach-back":
            # Step 1: run multi-reachability backwards in time
            self._run_hj_reachability(
                initial_values=self.get_initial_values(direction="multi-time-reach-back"),
                t_start=x_t.date_time,
                T_max_in_seconds=self.specific_settings["T_goal_in_seconds"],
                dir="multi-time-reach-back",
            )

            # Now just extract it forwards releasing the vehicle at t=0
            def termination_condn(x_target, r, x, t):
                return np.linalg.norm(x_target - x) <= r

            termination_condn = partial(
                termination_condn,
                jnp.array(self.problem.end_region.__array__()),
                self.problem.target_radius,
            )
            self._extract_trajectory(
                self.get_x_from_full_state(x_t), termination_condn=termination_condn
            )
            # arrange to forward times by convention for plotting and open-loop control (aka closed-loop with this)
            self._flip_value_func_to_forward_times()
            if self.all_values.min() < -2:
                raise ValueError(
                    "HJPlanner: Some issue with the value function, min goes below -2, should maximally be -1."
                )
        else:
            raise ValueError(
                "Direction in controller YAML needs to be one of {backward, forward, forward-backward, "
                "multi-time-reach-back}"
            )

        # check if all_values contains any Nans
        if jnp.isnan(self.all_values).sum() > 0:
            raise ValueError(
                "HJ Planner has NaNs in all values. Something went wrong in solving the PDE."
            )

        self.last_planning_posix = x_t.date_time.timestamp()



    def _update_current_data(self, observation: ArenaObservation):
        """Helper function to load new current data into the interpolation.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        """
        start = time.time()

        # Step 1: get the x,y,t bounds for current position, goal position and settings.
        t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
            x_0=observation.platform_state.to_spatio_temporal_point(),
            x_T=self.problem.end_region,
            deg_around_x0_xT_box=self.specific_settings["deg_around_xt_xT_box"],
            temp_horizon_in_s=self.specific_settings["T_goal_in_seconds"],
        )
        # adjust if specified explicitly in settings
        if "x_interval" in self.specific_settings:
            x_interval = self.specific_settings["x_interval"]
        if "y_interval" in self.specific_settings:
            y_interval = self.specific_settings["y_interval"]

        # get the data subset from the file
        data_xarray = observation.forecast_data_source.get_data_over_area(
            x_interval=x_interval,
            y_interval=y_interval,
            t_interval=t_interval,
            spatial_resolution=self.specific_settings["grid_res"],
        )

        # get seaweed growth rate data subset from analytical function as xarray
        path = "./data/seaweed/seaweed_precomputed_over_area.nc"
        if os.path.exists(path):
            seaweed_xarray = xr.open_dataset(path)
        else:
            seaweed_xarray = self.arena.seaweed_field.hindcast_data_source.get_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                t_interval=t_interval,
                spatial_resolution=self.specific_settings["grid_res"] * 2,
            )
            seaweed_xarray.to_netcdf(path=path)

        # calculate relative posix_time (we use it in interpolation because jax uses float32 and otherwise cuts off)
        data_xarray = data_xarray.assign(
            relative_time=lambda x: units.get_posix_time_from_np64(x.time)
            - units.get_posix_time_from_np64(data_xarray["time"][0])
        )
        seaweed_xarray = seaweed_xarray.assign(
            relative_time=lambda x: units.get_posix_time_from_np64(x.time)
            - units.get_posix_time_from_np64(seaweed_xarray["time"][0])
        )

        # feed in the current data to the Platform classes
        self.nondim_dynamics.dimensional_dynamics.update_jax_interpolant(
            data_xarray, seaweed_xarray
        )

        # set absolute time in UTC Posix time
        self.current_data_t_0 = units.get_posix_time_from_np64(data_xarray["time"][0]).data
        # set absolute final time in UTC Posix time
        self.current_data_t_T = units.get_posix_time_from_np64(data_xarray["time"][-1]).data

        # initialize the grids and dynamics to solve the PDE with
        self.initialize_hj_grid(data_xarray)
        self._initialize_non_dim_grid()
        # update non_dimensional_dynamics with the new non_dim scaling and offset
        self.nondim_dynamics.characteristic_vec = self.characteristic_vec
        self.nondim_dynamics.offset_vec = self.offset_vec

        # Delete the old caches (might not be necessary for analytical fields -> investigate)
        self.logger.debug("HJPlannerBase: Cache Size " + str(hj.solver._solve._cache_size()))
        hj.solver._solve._clear_cache()
        # xla._xla_callable.cache_clear()

        # For now only using interpolation in jnp (no analytical function)

        # Option 1: Data Source is an analytical field
        # if self.forecast_data_source['data_source_type'] == 'analytical_function':
        #     # calculate target shape of the grid
        #     x_n_res = int((lon_bnds[-1] - lon_bnds[0]) / self.specific_settings['grid_res'][0])
        #     y_n_res = int((lat_bnds[-1] - lat_bnds[0]) / self.specific_settings['grid_res'][1])
        #
        #     # get the grid dict
        #     grids_dict, _ = self.forecast_data_source['content'].get_grid_dict(
        #         t_interval, lat_interval=lat_bnds, lon_interval=lon_bnds, spatial_shape=(x_n_res, y_n_res))
        #     grids_dict['not_plot_land'] = True
        #
        #     self.nondim_dynamics.dimensional_dynamics.set_currents_from_analytical(self.forecast_data_source)
        #     self.forecast_data_source['content'].current_run_t_0 = grids_dict['t_grid'][0]

        self.logger.info(f"HJPlannerBase: Loading new Current Data ({time.time() - start:.1f}s)")

    