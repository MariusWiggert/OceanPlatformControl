import os
import pickle
import time
from typing import Dict, Optional, Union

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
import scipy
import xarray as xr
from scipy.interpolate import interp1d

from ocean_navigation_simulator.controllers.hj_planners.HJPlannerBaseDim import (
    HJPlannerBaseDim,
)
from ocean_navigation_simulator.controllers.hj_planners.Platform2dSeaweedForSim import (
    Platform2dSeaweedForSim,
)
from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.utils import units


class HJBSeaweed2DPlanner(HJPlannerBaseDim):
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
        self.specific_settings["platform_dict"] = problem.platform_dict

        # set first_plan to True so we plan on the first run over the whole time horizon
        self.first_plan = True
        (
            self.reach_times_global,
            self.all_values_global,
            self.all_values_subset,
            self.grid_global,
        ) = [None] * 4

    def get_x_from_full_state(
        self, x: Union[PlatformState, SpatioTemporalPoint, SpatialPoint]
    ) -> jnp.ndarray:
        return jnp.array(x.__array__())[:2]

    def get_time_vector(self, T_max_in_seconds: int) -> int:
        """Return n_time_vector for a given T_max_in_seconds. If we plan over the full horizon we take complete n_time_vector. If we only replan the forecast horizon and take the previous value fct. as initial values we shorten the n_time_vector accordingly"""
        return np.rint(
            (T_max_in_seconds * self.specific_settings["n_time_vector"])
            / self.specific_settings["T_goal_in_seconds"]
        ).astype(int)

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
            return jnp.zeros(self.grid.shape)
        elif direction == "backward":
            return jnp.zeros(self.grid.shape)
        elif direction == "multi-time-reach-back":
            raise NotImplementedError("HJPlanner: Multi-Time-Reach not implemented yet")
        else:
            raise ValueError(
                "Direction in specific_settings of HJPlanner needs to be forward, backward, or multi-reach-back."
            )

    def _get_idx_closest_value_in_array(self, array: np.ndarray, value: Union[int, float]) -> int:
        """Takes a value and an array and returns the index of the closest value in the array.
        Args:
            array: array in which to find the index of the closest value
            value: value for which to find closest array entry / index
        Returns:
            idx: index of closest value in array
        """
        return np.argmin(abs(array - value))

    def set_subset_interpolator(self):
        """Helper Function to create an interpolator for the value function for retrieving interpolated subsets and fast computation"""

        self.reach_times_global_flipped, self.all_values_global_flipped = [
            np.flip(seq, axis=0) for seq in [self.reach_times_global, self.all_values_global]
        ]
        self.subset_interpolator = scipy.interpolate.RegularGridInterpolator(
            points=(
                self.reach_times_global_flipped,
                self.grid_global.coordinate_vectors[0],
                self.grid_global.coordinate_vectors[1],
            ),
            values=self.all_values_global_flipped,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

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
            #  # log values for closed-loop trajectory extraction
            # x_start_backtracking = self.get_x_from_full_state(self.problem.end_region)
            # t_start_backtracking = (
            #     x_t.date_time.timestamp() + self.specific_settings["T_goal_in_seconds"]
            # )

        elif self.specific_settings["direction"] == "backward":
            # Note: no trajectory is extracted as the value function is used for closed-loop control

            # Check whether we plan the first time over the whole time-horizon or only over days with new forecast and recycle value fct. for the remaining days
            if self.first_plan:
                initial_values = self.get_initial_values(direction="backward")
                T_max_in_seconds = self.specific_settings["T_goal_in_seconds"]
            elif (
                not self.first_plan
                and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
            ):
                # Get index of closest global reach time for the end of the forecast horizon
                time_idx = self._get_idx_closest_value_in_array(
                    self.reach_times_global, self.forecast_from_start
                )

                # Interpolation:
                # Create 3D arrays for the coordinates
                T, LON, LAT = np.meshgrid(
                    self.reach_times_global_flipped,
                    self.grid.states[:, 0, 0],
                    self.grid.states[0, :, 1],
                    indexing="ij",
                )
                # Flatten the arrays into lists
                coords = np.stack((T.flatten(), LON.flatten(), LAT.flatten()), axis=1)

                # Call the interpolation function
                interpolated = self.subset_interpolator(coords)

                # Reshape the result into a 3D array
                self.all_values_subset_flipped = interpolated.reshape(
                    (
                        len(self.reach_times_global_flipped),
                        len(self.grid.states[:, 0, 0]),
                        len(self.grid.states[0, :, 1]),
                    )
                )

                # Flip from forward time to backward time
                self.all_values_subset = np.flip(self.all_values_subset_flipped, axis=0)

                # Get value function at end of FC Horizon
                # initial_values = self.all_values_subset[-time_idx]
                initial_values = interp1d(
                    self.reach_times_global, self.all_values_subset, axis=0, kind="linear"
                )(self.forecast_from_start).squeeze()

                # Get T_max only for FC - so replanning only runs over this timeframe
                T_max_in_seconds = int(self.forecast_length)

            self._run_hj_reachability(
                initial_values=initial_values,
                t_start=x_t.date_time,
                T_max_in_seconds=T_max_in_seconds,
                dir="backward",
            )

            if (
                self.first_plan
                and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
            ):
                # Set first_plan to False after first planning is finished
                self.first_plan = False

                # Save global value fct., reach times & grid for later re-use
                self.all_values_global = self.all_values
                self.reach_times_global = self.reach_times
                self.grid_global = self.grid

                # Save initial start time
                self.x_0_time = np.datetime64(x_t.date_time, "ns")

                # Set value fct. subset interpolator
                self.set_subset_interpolator()

            elif (
                not self.first_plan
                and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
            ):
                self.logger.info("HJBSeaweed2DPlanner: concatenate pre-computed and new value fct.")
                # Concatenate the the static part and the new part of the value fct. based on new FC data
                # Shift global reach times to account for temporal progress
                time_progress = self.current_data_t_0 - units.get_posix_time_from_np64(
                    self.x_0_time
                )
                self.reach_times = jnp.concatenate(
                    [self.reach_times_global[:time_idx] - time_progress, self.reach_times], axis=0
                )
                self.all_values = jnp.concatenate(
                    [self.all_values_subset[:time_idx], self.all_values], axis=0
                )

            # arrange to forward times by convention for plotting and open-loop control
            self._set_value_func_to_forward_time()
            # log values for closed-loop trajectory extraction
            x_start_backtracking = self.get_x_from_full_state(x_t)
            t_start_backtracking = x_t.date_time.timestamp()

        elif self.specific_settings["direction"] == "forward-backward":
            raise NotImplementedError("HJPlanner: Forward-Backward not implemented yet")
        elif self.specific_settings["direction"] == "multi-time-reach-back":
            raise NotImplementedError("HJPlanner: Multi-Time-Reach not implemented yet")
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

        # extract trajectory for open_loop control or because of plotting/logging
        if (
            self.specific_settings["direction"] == "forward"
            or self.specific_settings.get("calc_opt_traj_after_planning", True)
            or not self.specific_settings.get("closed_loop", True)
        ):
            self.times, self.x_traj, self.contr_seq, self.distr_seq = self._extract_trajectory(
                x_start=x_start_backtracking,
                t_start=t_start_backtracking,
                num_traj_disc=self.specific_settings.get("num_traj_disc", None),
                dt_in_sec=self.specific_settings.get("traj_dt_in_sec", None),
            )
            self._log_traj_in_plan_dict(self.times, self.x_traj, self.contr_seq)

        self.last_planning_posix = x_t.date_time.timestamp()

    def _update_current_data(self, observation: ArenaObservation):
        """Helper function to load new current data into the interpolation.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        """
        start = time.time()

        # Extract FC length in seconds -> if else in order to also work with toy examples i.e current highway
        # TODO. change to posix time
        if hasattr(observation.forecast_data_source, "forecast_data_source"):
            self.forecast_length = (
                observation.forecast_data_source.forecast_data_source.forecast_data_source.DataArray.time.max()
                - np.datetime64(observation.platform_state.date_time, "ns")
            ) / np.timedelta64(1, "s")
        else:
            self.forecast_length = 3600 * 24 * 10

        if self.first_plan and self.forecast_length < self.specific_settings["T_goal_in_seconds"]:
            deg_around_x0_xT_box = self.specific_settings["deg_around_xt_xT_box_global"]
        else:
            deg_around_x0_xT_box = self.specific_settings["deg_around_xt_xT_box"]

        if (
            not self.first_plan
            and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
        ):
            # Extract relative FC Horizon from inital time
            self.forecast_from_start = (
                units.get_posix_time_from_np64(
                    observation.forecast_data_source.forecast_data_source.forecast_data_source.DataArray.time.max()
                )
                - units.get_posix_time_from_np64(self.x_0_time)
            ).data
            # Make sure forecast horizon is not longer as planning period
            if self.forecast_from_start > self.specific_settings["T_goal_in_seconds"]:
                self.forecast_from_start = self.specific_settings["T_goal_in_seconds"]

        # Step 1: get the x,y,t bounds for current position, goal position and settings.
        t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
            x_0=observation.platform_state.to_spatio_temporal_point(),
            x_T=observation.platform_state.to_spatio_temporal_point(),
            deg_around_x0_xT_box=deg_around_x0_xT_box,
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

        seaweed_xarray = self.arena.seaweed_field.hindcast_data_source.get_data_over_area(
            x_interval=x_interval,
            y_interval=y_interval,
            t_interval=t_interval,
            spatial_resolution=self.specific_settings["grid_res"],
        )

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
        self.dim_dynamics.update_jax_interpolant(data_xarray, seaweed_xarray)

        # set absolute time in UTC Posix time
        self.current_data_t_0 = units.get_posix_time_from_np64(data_xarray["time"][0]).data
        # set absolute final time in UTC Posix time
        self.current_data_t_T = units.get_posix_time_from_np64(data_xarray["time"][-1]).data

        # initialize the grids and dynamics to solve the PDE with
        self.initialize_hj_grid(data_xarray)

        # import plotly.graph_objects as go
        # import numpy as np

        # # Read data from a csv
        # z = seaweed_xarray["F_NGR_per_second"].data[-1]
        # x = self.grid.states[..., 0]
        # y = self.grid.states[..., 1]
        # fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        # # fig.update_layout(title='Mt Bruno Elevation', autosize=False,
        # #                   width=500, height=500,
        # #                   margin=dict(l=65, r=50, b=65, t=90))
        # fig.show()

        # update non_dimensional_dynamics with the new non_dim scaling and offset

        # Delete the old caches (might not be necessary for analytical fields -> investigate)
        self.logger.debug("HJBSeaweed2DPlanner: Cache Size " + str(hj.solver._solve._cache_size()))
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

        self.logger.info(
            f"HJBSeaweed2DPlanner: Loading new Current Data ({time.time() - start:.1f}s)"
        )

    def save_planner_state(self, folder):
        os.makedirs(folder, exist_ok=True)
        # Settings
        with open(folder + "specific_settings.pickle", "wb") as file:
            pickle.dump(self.specific_settings, file)
        # Used in Replanning
        # with open(folder + "last_fmrc_idx_planned_with.pickle", "wb") as file:
        #     pickle.dump(self.last_fmrc_idx_planned_with, file)
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
        with open(folder + "initial_values.pickle", "wb") as file:
            pickle.dump(self.initial_values, file)

    @staticmethod
    def from_saved_planner_state(
        folder, problem: NavigationProblem, arena: ArenaFactory, verbose: Optional[int] = 0
    ):
        # Settings
        with open(folder + "specific_settings.pickle", "rb") as file:
            specific_settings = pickle.load(file)

        planner = HJBSeaweed2DPlanner(
            problem=problem, specific_settings=specific_settings, arena=arena
        )

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
