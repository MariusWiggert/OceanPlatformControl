import os
import pickle
import shutil
import time
from datetime import datetime, timedelta, timezone
from typing import AnyStr, Callable, Dict, List, Optional, Tuple, Union

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
import scipy
import xarray as xr
from c3python import C3Python
from matplotlib import pyplot as plt
from matplotlib.animation import Animation
from scipy.interpolate import interp1d
from tqdm import tqdm

import ocean_navigation_simulator
from ocean_navigation_simulator.controllers.hj_planners.HJPlannerBaseDim import (
    HJPlannerBaseDim,
)
from ocean_navigation_simulator.controllers.hj_planners.Platform2dSeaweedForSim import (
    Platform2dSeaweedForSim,
)
from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.environment.SeaweedProblem import SeaweedProblem
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import get_c3, timing_logger


class CorruptedPreComputedValueFcts(Exception):
    def __repr__(self):
        return str(self)


class HJBSeaweed2DPlanner(HJPlannerBaseDim):
    """Reachability planner for 2D (lat, lon) reachability computation."""

    # TODO: describe five cases which can be handeled (combination of : if Precomput. and which datasources used and horizon)

    gpus: float = 1.0

    def __init__(
        self,
        arena: ArenaFactory,
        problem: SeaweedProblem,
        specific_settings: Optional[Dict] = ...,
    ):

        # get arena object for accessing seaweed growth model
        self.arena = arena
        super().__init__(problem, specific_settings)
        self.specific_settings = {
            "platform_dict": problem.platform_dict if problem is not None else None,
            "grid_res": 0.083,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
            "deg_around_xt_xT_box": 1,  # Area over which to run HJ_reachability
            "precomputation": False,  # Defines whether value fct. should be precomputed or normal planning is requested
            "value_fct_folder": "temp/precomputed_value_fcts/",  # Where to save and load precomputed value fcts
            "precomputed_local": False,  # Specify whether value fct are already downloaded or not
        } | self.specific_settings

        (
            self.all_values_subset,
            self.all_values_global_flipped,
            self.reach_times_global_flipped_posix,
            self.x_grid_global,
            self.y_grid_global,
        ) = [None] * 5
        self.hindcast_as_forecast = False

    def get_x_from_full_state(
        self, x: Union[PlatformState, SpatioTemporalPoint, SpatialPoint]
    ) -> jnp.ndarray:
        return jnp.array(x.__array__())[:2]

    # def get_time_vector(self, T_max_in_seconds: int) -> int:
    #     """Return n_time_vector for a given T_max_in_seconds. If we plan over the full horizon we take complete n_time_vector. If we only replan the forecast horizon and take the previous value fct. as initial values we shorten the n_time_vector accordingly"""
    #     return np.rint(
    #         (T_max_in_seconds * self.specific_settings["n_time_vector"])
    #         / self.specific_settings["T_goal_in_seconds"]
    #     ).astype(int)

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

    def _dirichlet(self, x, pad_width: int):
        """Dirichlet boundry conditions for PDE solve"""
        return jnp.pad(x, ((pad_width, pad_width)), "constant", constant_values=1)

    def initialize_hj_grid(self, xarray: xr) -> None:
        """Initialize the dimensional grid in degrees lat, lon"""
        # initialize grid using the grids_dict x-y shape as shape with dirichlet boundry conditions
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj.sets.Box(
                lo=np.array([xarray["lon"][0].item(), xarray["lat"][0].item()]),
                hi=np.array([xarray["lon"][-1].item(), xarray["lat"][-1].item()]),
            ),
            boundary_conditions=(self._dirichlet, self._dirichlet),
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
            # Check whether we plan the first time in order to retrieve the gloabl value fct for the time period after the interval we have FC data available.
            # Only in case FC horizon is shorter than acutal planning horizon --> otherwise we wouldn't need the averages
            if (
                not self.specific_settings["precomputation"]
                and self.all_values_global_flipped is None
                and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
            ):
                self.logger.info("HJBSeaweed2DPlanner: get pre-computed value fct.")
                # initial_values = self.get_initial_values(direction="backward")
                start_time = x_t.date_time + timedelta(seconds=self.forecast_length)
                end_time = x_t.date_time + timedelta(
                    seconds=self.specific_settings["T_goal_in_seconds"]
                )
                # TODO: Figure out way to get dataSource from arenaConfig instead of defining it manually in specific settings
                (
                    self.all_values_global_flipped,
                    self.reach_times_global_flipped_posix,
                    self.x_grid_global,
                    self.y_grid_global,
                ) = self._get_value_fct_reach_times(
                    t_interval=[start_time, end_time],
                    u_max=self.specific_settings["platform_dict"]["u_max_in_mps"],
                    dataSource=self.specific_settings["dataSource"],
                )

            if (
                not self.specific_settings["precomputation"]
                and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
            ):
                # Plan over days with new forecast and recycle gloabl value fct. for the remaining days
                # Interpolate subset from global value fct.
                self.logger.info(
                    "HJBSeaweed2DPlanner: get interpolate subset from pre-computed value fct."
                )

                # Create an interpolator for the value function for retrieving interpolated subsets and for fast computation
                self.subset_interpolator = scipy.interpolate.RegularGridInterpolator(
                    points=(
                        self.reach_times_global_flipped_posix,
                        self.x_grid_global,
                        self.y_grid_global,
                    ),
                    values=self.all_values_global_flipped,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
                # Create 3D arrays for the coordinates
                T, LON, LAT = np.meshgrid(
                    self.reach_times_global_flipped_posix,
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
                        len(self.reach_times_global_flipped_posix),
                        len(self.grid.states[:, 0, 0]),
                        len(self.grid.states[0, :, 1]),
                    )
                )

                # Flip from forward time to backward time
                self.all_values_subset = np.flip(self.all_values_subset_flipped, axis=0)
                self.reach_times_global_posix = np.flip(
                    self.reach_times_global_flipped_posix, axis=0
                )
                # Extract forecast end time in posix
                self.forecast_end_time_posix = x_t.date_time.timestamp() + self.forecast_length

                # Get index of closest global reach time for the end of the forecast horizon
                time_idx = self._get_idx_closest_value_in_array(
                    self.reach_times_global_posix, self.forecast_end_time_posix
                )

                # Get value function at end of FC Horizon to initialize backward HJ comp. within FC horizon
                # initial_values = self.all_values_subset[time_idx]
                initial_values = interp1d(
                    self.reach_times_global_posix, self.all_values_subset, axis=0, kind="linear"
                )(self.forecast_end_time_posix).squeeze()

                # Get T_max only for FC - so planning only runs over this time interval
                T_max_in_seconds = int(self.forecast_length)

            else:
                # In case FC horizon is longer than actual planning horizon --> we don't need the averages then
                # Or if we want to precompute a value fct.
                initial_values = self.get_initial_values(direction="backward")
                T_max_in_seconds = self.specific_settings["T_goal_in_seconds"]

            # Only run HJ if we do NOT compute on pure HC data or we precomputate the value fct.
            if not self.hindcast_as_forecast or self.specific_settings["precomputation"]:
                self._run_hj_reachability(
                    initial_values=initial_values,
                    t_start=x_t.date_time,
                    T_max_in_seconds=T_max_in_seconds,
                    dir="backward",
                )

            # Concatenate the the static part and the new part of the value fct. based on new FC data
            if (
                not self.specific_settings["precomputation"]
                and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
                and not self.hindcast_as_forecast
            ):
                self.logger.info("HJBSeaweed2DPlanner: concatenate pre-computed and new value fct.")
                # Shift global reach times to account for temporal progress
                # TODO: add temporal step in between reach times, right now the will be two same values on the alignment of both arrays
                reach_times_global_subset = (
                    self.reach_times_global_posix[
                        :time_idx
                    ]  # Get until end of FC horizon without last element which is already contained in self.reach_times
                    - self.reach_times_global_posix[time_idx]  # Shift to get relative POSIX time
                    + self.reach_times[0]  # Shift s.t. arrays will be continuous when concatenated
                )
                self.reach_times = jnp.concatenate(
                    [
                        reach_times_global_subset,
                        self.reach_times,
                    ],
                    axis=0,
                )
                self.all_values = jnp.concatenate(
                    [self.all_values_subset[:time_idx], self.all_values], axis=0
                )

            # Assign the values and reach times over the complete planning horizon if we only compute on pure HC data if we use precomputed values
            if self.hindcast_as_forecast and not self.specific_settings["precomputation"]:
                self.all_values = self.all_values_subset
                # Shift to get relative POSIX time
                self.reach_times = self.reach_times_global_posix - self.reach_times_global_posix[-1]
                # Set current_data_t_0 to first forward reach time value since it will be used later on to convert back to POSIX and our data_array is not used.
                self.current_data_t_0 = self.reach_times_global_posix[-1]

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

        # TODO: check if forecast_data_source.forecast_data_source still necessary - probably not!
        # Extract FC length in seconds -> if else in order to also work with toy examples i.e current highway
        # TODO: change to posix time
        if isinstance(
            observation.forecast_data_source.forecast_data_source,
            ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource.ForecastFromHindcastSource,
        ):
            self.forecast_length = (
                observation.forecast_data_source.forecast_data_source.forecast_length_in_days
                * 3600
                * 24
            )

        elif hasattr(observation.forecast_data_source, "forecast_data_source") and not isinstance(
            observation.forecast_data_source.forecast_data_source,
            ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource.HindcastFileSource,
        ):
            self.forecast_length = int(
                (
                    observation.forecast_data_source.forecast_data_source.DataArray.time.max()
                    - np.datetime64(observation.platform_state.date_time, "ns")
                )
                / np.timedelta64(1, "s")
            )
        else:

            self.forecast_length = 0

            # If FC source is HC Hycom or Copernicus and not averages
            if not self.specific_settings["dataSource"] == "average":
                self.hindcast_as_forecast = True

        # Decide how much temporal data we want to load to our data arrays
        if self.specific_settings["precomputation"]:
            # If we precompute the value fct. we need data over our complete horizon
            temp_horizon_in_s = self.specific_settings["T_goal_in_seconds"]
        else:
            # If we do not precompute we only need the data for the the forecast horizon since everything else is already covered in our precomputed values
            # Note: If our T_goal_in_seconds is smaller than our forecast horizon we actually load more data than needed - but shouldn't be too much overload and keeps the code cleaner
            temp_horizon_in_s = self.forecast_length

        # Step 1: get the x,y,t bounds for current position, goal position and settings.
        t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
            x_0=observation.platform_state.to_spatio_temporal_point(),
            x_T=observation.platform_state.to_spatio_temporal_point(),
            deg_around_x0_xT_box=self.specific_settings["deg_around_xt_xT_box"],
            temp_horizon_in_s=temp_horizon_in_s,
        )
        # adjust if specified explicitly in settings
        if "x_interval" in self.specific_settings:
            x_interval = self.specific_settings["x_interval"]
        if "y_interval" in self.specific_settings:
            y_interval = self.specific_settings["y_interval"]

        grid_res = self.specific_settings["grid_res"]
        dataSource = self.specific_settings.get("dataSource")

        # Step 1: Sanitize Inputs
        if self.specific_settings.get("precomputation") and dataSource is None:
            raise KeyError("A dataSource must be provided in specific_settings for precomputation.")
        if dataSource is not None and dataSource.lower() not in [
            "hc_hycom",
            "hc_copernicus",
            "average",
        ]:
            raise ValueError(
                f"dataSource {dataSource} invalid choose from: HC_HYCOM, HC_Copernicus and average."
            )

        # get the data subset from the file
        if not self.specific_settings["precomputation"]:
            # Take forecast data if we do not precompute the value function
            data_xarray = observation.forecast_data_source.get_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                t_interval=t_interval,
                spatial_resolution=grid_res,
            )
        elif (
            self.specific_settings["precomputation"]
            and self.specific_settings["dataSource"].lower() == "hc_hycom"
            or self.specific_settings["dataSource"].lower() == "hc_copernicus"
        ):
            # Take also forecast data if we need to precompute the value function on HC data. We take the forecast data since if there is no forecast config provided in the arena config the hindcast is taken as forecast.
            data_xarray = observation.forecast_data_source.get_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                t_interval=t_interval,
                spatial_resolution=grid_res,
            )
        elif (
            self.specific_settings["precomputation"]
            and self.specific_settings["dataSource"].lower() == "average"
        ):
            # Take the average data if we need to precompute the value function on average data.
            data_xarray = observation.average_data_source.get_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                t_interval=t_interval,
                temporal_resolution=7200,  # TODO: add configurable temporal_resolution
                spatial_resolution=grid_res,
            )

            # reduce temporal margins since the averages will be returned with large temporal margins (+- 1 month)
            dt = data_xarray["time"][1] - data_xarray["time"][0]
            data_xarray = data_xarray.sel(
                time=slice(
                    np.datetime64(t_interval[0].replace(tzinfo=None)) - dt,
                    np.datetime64(t_interval[1].replace(tzinfo=None)) + dt,
                )
            )

        seaweed_xarray = self.arena.seaweed_field.hindcast_data_source.get_data_over_area(
            x_interval=x_interval,
            y_interval=y_interval,
            t_interval=t_interval,
            spatial_resolution=grid_res,
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

    def _get_value_fct_reach_times(
        self, t_interval: List[datetime], u_max: float, dataSource: str
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        # TODO: extend to get different Regions. right now only Region Matthias
        """Returns value fct. and reach times for a given time interval
        Args:
            t_interval: List of start and end time as datetime objects.
            u_max:      Maximum control in m/s
            dataSource: Defines on which source the value fct was computed on
        Returns:
            value_fct_array: Concatenated value fct. array where values get adjusted accordingly.
        """

        hj_val_func_list = self._fetch_hj_val_func_list_list_from_c3_db(
            t_interval=t_interval, u_max=u_max, dataSource=dataSource
        )
        reach_times_list = [hj_val_func["time"].data for hj_val_func in hj_val_func_list]
        value_fct_list = [hj_val_func["HJValueFunc"].data for hj_val_func in hj_val_func_list]

        # Concatenate the different value functions and reach times in one array
        value_fct_array = self._concatenate_value_fcts(value_fct_list=value_fct_list)
        reach_times_array = self._concatenate_reach_times(reach_times_list=reach_times_list)
        # Get the indices for the start and end times since value_fct_array and reach_times_array will have a larger interval than needed
        idx_t_interval = self._get_idx_time_interval_in_reach_times(
            reach_times_array=reach_times_array, t_interval=t_interval
        )

        # Get correct subset of value fct. over the requested time interval. Substract last value to imitate initializiation with zero as last value.
        value_fct_array = (
            value_fct_array[idx_t_interval[0] : idx_t_interval[1] + 1]
            - value_fct_array[idx_t_interval[1]]
        )
        reach_times_array = reach_times_array[idx_t_interval[0] : idx_t_interval[1] + 1]

        # if(config["forecast"] is not None):
        #     #TBDONE only return earliest value and at which time

        return (
            value_fct_array,
            reach_times_array,
            hj_val_func_list[0]["x"].data,
            hj_val_func_list[0]["y"].data,
        )

    def _concatenate_value_fcts(self, value_fct_list: List) -> np.array:
        """Concatenate the value_fcts in the list by stacking and adding them up
        Args:
            value_fct_list: List of all value functions corresponding to the requested time interval. Ordered in forward time.
        Returns:
            value_fct_array: Concatenated value fct. array where values get adjusted accordingly.
        """
        # Get first value of last pre-computed v. fct. in list and take it as init value
        init_value = value_fct_list[-1][0]

        # Reverse the value fct. list in place
        value_fct_list.reverse()

        # Loop over reverse v. fct. list excl. latest entry of the v. fct. list
        for value_fct in value_fct_list[1:]:
            # Add the first value of the next v. fct. to the value fct.
            value_fct += init_value
            # Get first value as inital value for next iteration
            init_value = value_fct[0]

        # Reverse the value fct. list again to restore its original order
        value_fct_list.reverse()

        return np.concatenate(value_fct_list)

    def _concatenate_reach_times(self, reach_times_list: List) -> np.array:
        """Concatenate the reach time in the list s.t. the times are strictly ascending.
        As the first reach time of one array can have the same value as the last time of the previous array.
        Args:
            reach_times_list: List of all reach times corresponding to the requested time interval. Ordered in forward time.
        Returns:
            reach_times_array: Concatenated reach times array.
        """

        for i, reach_times in enumerate(reach_times_list[:-1]):
            # Check if concatenation would preserve strict ascending of time
            # When shifting the relative reach times with self.current_data_t_0 in order to retrieve POSIX time we sometimes get issues if values are too close together or the same value.
            # This is due to rounding when later on the arrays are transformed to JAX float32 arrays.
            # Step 1: check if last value of curr. array and first of next one are at least 200 seconds apart.
            if reach_times_list[i + 1][0] - reach_times[-1] < 200:
                # Substract 200 seconds from last time to enforce strict ascending even after rounding. Should not change the overall behaviour since dt is usually 600 seconds.
                reach_times[-1] -= 200
                self.logger.debug(
                    "HJBSeaweed2DPlanner: Adjusted reach times while concatenating in order to make times strictly ascending."
                )
            # Step 2: check if last value of curr. array is larger than first of next one
            elif reach_times[-1] > reach_times_list[i + 1][0]:
                raise ValueError(
                    f"HJBSeaweed2DPlanner: Failed to concatenate reach times since they are not ascending. {reach_times[-1]} is larger than next value: {reach_times_list[i + 1][0]}"
                )

        return np.concatenate(reach_times_list)

    def _fetch_hj_val_func_list_list_from_c3_db(
        self, t_interval: List[datetime], u_max: float, dataSource: str
    ) -> List:
        """Should get a list of xarrays (with value_fcts and corresponding reach_times (in sec) and x_grid, y_grid)
        Args:
            t_interval: List of start and end time as datetime objects.
            u_max:      Maximum control in m/s
            dataSource: Defines on which source the value fct was computed on
        Returns:
            hj_val_func_list: List of value_fcts and corresponding reach_times (in sec) and x_grid, y_grid
        """

        with timing_logger(
            "Download pre-computed value functions: {start} until {end} ({{}})".format(
                start=t_interval[0].strftime("%Y-%m-%d %H-%M-%S"),
                end=t_interval[-1].strftime("%Y-%m-%d %H-%M-%S"),
            ),
            self.logger,
        ):

            files = self._download_required_files(
                dataSource=dataSource,
                u_max=u_max,
                download_folder=self.specific_settings["value_fct_folder"],
                t_interval=t_interval,
            )

            self.logger.debug(f"HJBSeaweed2DPlanner: Value Fct Files: {files}")

            hj_val_func_list = []
            for file in files:
                hj_val_func_list.append(xr.open_dataset(file))

            return hj_val_func_list

    def _get_idx_time_interval_in_reach_times(
        self, reach_times_array: np.array, t_interval: List[datetime]
    ) -> List:
        """Takes reach times and a time interval and returns the index for starta and end of the closest values in the reach times array.
        Args:
            reach_times_array: reach times array in seconds
            t_interval: List of start and end time as datetime objects for which to find closest array entry / index
        Returns:
            idxs: List of start and end index of closest value in array
        """
        # Get closest idx for start & end time
        start_idx = np.argmin(abs(reach_times_array - t_interval[0].timestamp()))
        end_idx = np.argmin(abs(reach_times_array - t_interval[1].timestamp()))
        return [start_idx, end_idx]

    def find_value_fct_files(path, t_interval):
        """Takes path to value fcts. and time interval and returns the corresponding file names.
        Args:
            path: path to folder with all value fcts.
            t_interval: List of start and end time as datetime objects.
        Returns:
            files: List of file names of relevant value fcts.
        """

        files = []

        start_min = t_interval[0]
        start_max = t_interval[1]

        for f in os.listdir(path):

            # Get date string from file name
            f_date_string = f[8:28]

            # Create time-aware datetime object from string
            start = datetime.strptime(f_date_string, "%Y-%m-%dT%H-%M-%SZ").replace(
                tzinfo=timezone.utc
            )

            if start_min <= start <= start_max:
                files.append(path + f)

        return files

    def _get_filelist(
        self,
        dataSource: str,
        u_max: float,
        t_interval: List[datetime] = None,
        c3: Optional[C3Python] = None,
    ):
        """
        helper function to get a list of available files from c3
        Args:
            dataSource: one of [HC_HYCOM, HC_Copernicus, average]
            u_max: float which defines the u_max used to precompute the value fcts.
            t_interval: List of datetime with length 2. None allows to search in all available times.
        Return:
            c3.FetchResult where objs contains the actual files
        """
        # Step 1: Sanitize Inputs
        if dataSource.lower() not in ["hc_hycom", "hc_copernicus", "average"]:
            raise ValueError(
                f"dataSource {dataSource} invalid choose from: HC_HYCOM, HC_Copernicus and average."
            )
        # Step 2: Find c3 type
        if c3 is None:
            c3 = get_c3()
        c3_file_type = getattr(c3, "PreComputedValueFct")

        # Step 3: Execute Query
        # Create strings for filters
        umax_filter = f"umax == {u_max}"
        source_filter = f' && dataSource == "{dataSource.lower()}"'
        if t_interval is not None:
            start_min = f"{t_interval[0]}"
            start_max = f"{t_interval[1]}"
            time_filter = f' && ( (timeCoverage.start <= "{start_min}" && "{start_min}" <= timeCoverage.end) || (timeCoverage.start <= "{start_max}" &&  "{start_max}" <= timeCoverage.end) || (timeCoverage.start >= "{start_min}" && timeCoverage.end <= "{start_max}"))'
        else:
            # accepting t_interval = None allows to download the whole file list for analysis
            time_filter = ""

        return c3_file_type.fetch(
            spec={
                "include": "[this]",  # TODO: check if it works like this
                "filter": umax_filter + source_filter + time_filter,
                "order": "ascending(timeCoverage.start)",
            }
        )

    def _download_filelist(
        self,
        files,
        download_folder,
        throw_exceptions,
        c3: Optional[C3Python] = None,
    ):
        """
        thread-safe download with corruption and file size check
        Arguments:
            files: c3.FetchResult object
            download_folder: folder to download files to
            throw_exceptions: if True throws exceptions for missing/corrupted files
            c3: c3 object
        Returns:
            list of downloaded files
        """
        if c3 is None:
            c3 = get_c3()

        self.logger.info(
            f"HJBSeaweed2DPlanner: Downloading {files.count} files to '{download_folder}'."
        )

        # Step 1: Sanitize Inputs
        if not download_folder.endswith("/"):
            download_folder += "/"
        os.makedirs(download_folder, exist_ok=True)

        # Step 2: Download Files thread-safe with atomic os.replace
        downloaded_files = []
        temp_folder = f"{download_folder}{os.getpid()}/"
        try:
            for file in tqdm(files.objs):
                filename = os.path.basename(file.valueFunction.contentLocation)
                url = file.valueFunction.url
                filesize = file.valueFunction.contentLength
                if (
                    not os.path.exists(download_folder + filename)
                    or os.path.getsize(download_folder + filename) != filesize
                ):
                    c3.Client.copyFilesToLocalClient(url, temp_folder)

                    error = False
                    # check file size match
                    if os.path.getsize(temp_folder + filename) != filesize:
                        error = "Incorrect file size ({filename}). Should be {filesize}B but is {actual_filesize}B.".format(
                            filename=filename,
                            filesize=filesize,
                            actual_filesize=os.path.getsize(download_folder + filename),
                        )
                    else:
                        # check valid xarray file and meta length
                        try:
                            f = xr.open_dataset(temp_folder + filename)
                            # TODO: check if really posix here
                            t_grid = f.variables["time"].data  # units.get_posix_time_from_np64()
                            # close netCDF file
                            f.close()
                            # create dict
                            grid_dict_list = {
                                "t_range": [
                                    datetime.fromtimestamp(t_grid[0], timezone.utc),
                                    datetime.fromtimestamp(t_grid[-1], timezone.utc),
                                ],
                            }

                            if (
                                file.timeCoverage.start < grid_dict_list["t_range"][0]
                                or grid_dict_list["t_range"][-1] < file.timeCoverage.end
                            ):
                                error = "File shorter than declared in meta: filename={filename}, meta: [{ms},{me}], file: [{gs},{ge}]".format(
                                    filename=filename,
                                    ms=file.timeCoverage.start.strftime("%Y-%m-%d %H-%M-%S"),
                                    me=file.timeCoverage.end.strftime("%Y-%m-%d %H-%M-%S"),
                                    gs=grid_dict_list["t_range"][0].strftime("%Y-%m-%d %H-%M-%S"),
                                    ge=grid_dict_list["t_range"][-1].strftime("%Y-%m-%d %H-%M-%S"),
                                )

                        except Exception:
                            error = f"HJBSeaweed2DPlanner: Corrupted file: '{filename}'."

                    if error and throw_exceptions:
                        raise CorruptedPreComputedValueFcts(error)
                    elif error:
                        os.remove(temp_folder + filename)
                        self.logger.warning(error)
                        continue

                    # Move thread-safe
                    os.replace(temp_folder + filename, download_folder + filename)
                    self.logger.info(
                        f"HJBSeaweed2DPlanner:  File downloaded: '{filename}', {filesize/10e6:.1f}MB."
                    )
                else:
                    # Path().touch()
                    os.system(f"touch {download_folder + filename}")
                    self.logger.info(
                        f"HJBSeaweed2DPlanner:  File already downloaded: '{filename}', {filesize/10e6:.1f}MB."
                    )

                downloaded_files.append(download_folder + filename)

        except BaseException:
            shutil.rmtree(temp_folder, ignore_errors=True)
            raise
        else:
            shutil.rmtree(temp_folder, ignore_errors=True)

        return downloaded_files

    def _download_required_files(
        self,
        dataSource: str,
        u_max: float,
        download_folder: str,
        t_interval: List[datetime],
        throw_exceptions: bool = False,
        c3: Optional[C3Python] = None,
    ) -> List[str]:
        """
        helper function for thread-safe download of available current files from c3
        Args:
            dataSource: one of [HC_HYCOM, HC_Copernicus, average]
            u_max: float which defines the u_max used to precompute the value fcts.
            download_folder: path on disk to download the files e.g. /tmp/value_fcts/
            t_interval: List of datetime with length 2.
            throw_exceptions: throw exceptions for missing or corrupted files
            c3: c3 object can be passed in directly, if not a c3 object is created
        Returns:
            list of downloaded files
        """
        # Step 1: Find relevant files
        files = self._get_filelist(dataSource=dataSource, u_max=u_max, t_interval=t_interval, c3=c3)

        # Step 2: Check File Count
        if files.count == 0:
            message = "No files in the database for {dataSource}, {u_max} and t_0={t_0} and t_T={t_T} ".format(
                dataSource=dataSource,
                u_max=u_max,
                t_0=t_interval[0],
                t_T=t_interval[1],
            )
            if throw_exceptions:
                raise CorruptedPreComputedValueFcts(message)
            else:
                self.logger.error(message)
                return 0

        # Step 3: Download files thread-safe
        downloaded_files = self._download_filelist(
            files=files,
            download_folder=download_folder,
            throw_exceptions=throw_exceptions,
            c3=c3,
        )

        return downloaded_files

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
        folder, problem: SeaweedProblem, arena: ArenaFactory, verbose: Optional[int] = 0
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

    def plot_value_fct_snapshot(
        self,
        rel_time_in_seconds: float = 0,
        ax: plt.Axes = None,
        return_ax: bool = False,
        fig_size_inches: Tuple[int, int] = (12, 12),
        alpha_color: float = 1.0,
        add_drawing: Callable[[plt.axis, float], None] = None,
        number_of_levels: int = 20,
        colorbar: bool = False,
        **kwargs,
    ):
        """Plot the value fct. the planner was computing last at a specific rel_time_in_seconds.
        Args:
            rel_time_in_seconds:    the relative time for which to plot the snapshot since last replan
            ax:                     Optional: axis object to plot on top of
            return_ax:              if true, function returns ax object for more plotting
            fig_size_inches:        Figure size
            alpha_color:            the alpha level of the color
            number_of_levels:       the number of levels/contours to plot
            colorbar:               defines if contour should be plotted as black lines (False) or colorized contor surfaces/areas (True)
            ### Other optional arguments
            add_drawing:            A callable to add a drawing to the snapshot, taking in (ax, rel_time_in_seconds)
        """
        if self.grid.ndim != 2:
            raise ValueError("plot_value_fct is currently only implemented for 2D sets")

        # create the axis object if not fed in
        if ax is None:
            if self.specific_settings["use_geographic_coordinate_system"]:
                ax = self.last_data_source.forecast_data_source.set_up_geographic_ax()
            else:
                ax = plt.axes()

        # interpolate the value function to the specific time
        val_at_t = interp1d(self.reach_times, self.all_values, axis=0, kind="linear")(
            max(
                self.reach_times[0],
                min(self.reach_times[-1], rel_time_in_seconds + self.reach_times[0]),
            )
        ).squeeze()

        vmin, vmax = val_at_t.min(), val_at_t.max()
        val_func_levels = np.linspace(vmin, vmax, number_of_levels)

        # package them in kwargs
        kwargs.update(
            {
                "val_func_levels": val_func_levels,
                "y_label": "HJ Value Function",
                # "yticklabels": abs_time_y_ticks,
            }
        )

        plot_level = val_func_levels if not colorbar else 0

        # plot the set on top of ax
        ax = hj.viz._visSet2D(
            grid=self.grid,
            data=val_at_t,
            plot_level=plot_level,
            color_level="black",
            colorbar=colorbar,
            obstacles=None,
            return_ax=True,
            input_ax=ax,
            alpha_colorbar=alpha_color,
            **kwargs,
        )

        ax.scatter(
            self.problem.start_state.lon.deg,
            self.problem.start_state.lat.deg,
            color="r",
            marker="o",
            zorder=6,
        )

        if self.specific_settings["use_geographic_coordinate_system"]:
            ax.set_title(
                "Value Function at time {}".format(
                    datetime.fromtimestamp(
                        self.reach_times[0] + rel_time_in_seconds + self.current_data_t_0,
                        tz=timezone.utc,
                    ).strftime("%Y-%m-%d %H:%M UTC")
                ),
                fontsize=20,
            )
        else:
            ax.set_title(
                "Value Function at time {} hours".format(
                    self.reach_times[0] + rel_time_in_seconds + self.current_data_t_0
                )
            )

        if add_drawing is not None:
            add_drawing(ax, rel_time_in_seconds)
        ax.set_facecolor("white")

        # adjust the fig_size
        fig = plt.gcf()
        fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
        if return_ax:
            return ax
        else:
            plt.show()

    def plot_value_fct_snapshot_over_currents(
        self, rel_time_in_seconds: float = 0, ax: plt.Axes = None, **kwargs
    ):
        """Plot the value fct. the planner was computing last at  a specific rel_time_in_seconds over the currents.
        Args:
            rel_time_in_seconds:    the relative time for which to plot the snapshot since last replan
            ax:                     Optional: axis object to plot on top of
            kwargs:                 See plot_value_fct_snapshot for further arguments
        """
        os.makedirs("generated_media", exist_ok=True)
        # plot currents on ax
        ax = self.last_data_source.forecast_data_source.plot_data_at_time_over_area(
            time=self.current_data_t_0 + rel_time_in_seconds,
            x_interval=[self.grid.domain.lo[0], self.grid.domain.hi[0]],
            y_interval=[self.grid.domain.lo[1], self.grid.domain.hi[1]],
            return_ax=True,
            colorbar=False,
            ax=ax,
        )
        # add reachability snapshot on top
        return self.plot_value_fct_snapshot(
            rel_time_in_seconds=rel_time_in_seconds,
            ax=ax,
            display_colorbar=True,
            **kwargs,
        )

    def plot_value_fct_animation(
        self,
        filefolder: AnyStr = "generated",
        filename: AnyStr = "reachability_animation.mp4",
        temporal_resolution: Optional[int] = None,
        spatial_resolution: Optional[float] = None,
        with_opt_ctrl: Optional[bool] = False,
        forward_time: Optional[bool] = False,
        data_source_for_plt: Optional = None,
        t_end: Optional[datetime] = None,
        fps: Optional[int] = 10,
        with_background: Optional[bool] = True,
        background_animation_args: Optional[dict] = {},
        number_of_levels: int = 20,
        **kwargs,
    ):
        """Create an animation of the value_fct computation.
        Args:
           filefolder:         specify to which folder aninamtions should be saved to - will be created if not existing
           filename:           filename under which to save the animation
           temporal_resolution: the temporal resolution in seconds, per default same as data_source
           with_opt_ctrl:      if True the optimal trajectory and control is added as overlay.
           forward_time:       forward_time manually force forward time (otherwise in direction of calculation)
           data_source_for_plt:the data source to plot as background with data_source.animate_data()
           kwargs:             See plot_value_fct_snapshot for further arguments (can also add drawings)
        """
        os.makedirs(filefolder, exist_ok=True)

        # interpolate the value function to the specific time
        vmin, vmax = self.all_values.min(), self.all_values.max()
        val_func_levels = np.linspace(vmin, vmax, number_of_levels)

        # package them in kwargs
        kwargs.update(
            {
                "val_func_levels": val_func_levels,
                "y_label": "HJ Value Function",
                # "yticklabels": abs_time_y_ticks,
            }
        )

        def add_value_fct_snapshot(ax, time):
            ax = self.plot_value_fct_snapshot(
                rel_time_in_seconds=time - self.current_data_t_0,
                alpha_color=1,
                return_ax=True,
                fig_size_inches=(12, 12),
                ax=ax,
                display_colorbar=True,
                number_of_levels=number_of_levels,
                **kwargs,
            )
            if with_opt_ctrl:
                # add the trajectory to it
                ax.plot(
                    self.x_traj[0, :],
                    self.x_traj[1, :],
                    color="black",
                    linewidth=2,
                    linestyle="--",
                    label="State Trajectory",
                )
                # get the planned idx of current time
                idx = np.searchsorted(a=self.times, v=time)
                # make sure it does not go over the array length
                idx = min(idx, len(self.times) - 2)
                # plot the control arrow for the specific time
                ax.scatter(
                    self.x_traj[0, idx], self.x_traj[1, idx], c="m", marker="o", s=20, zorder=9
                )
                ax.quiver(
                    self.x_traj[0, idx],
                    self.x_traj[1, idx],
                    self.contr_seq[0, idx] * np.cos(self.contr_seq[1, idx]),  # u_vector
                    self.contr_seq[0, idx] * np.sin(self.contr_seq[1, idx]),  # v_vector
                    color="magenta",
                    scale=10,
                    label="Control",
                    zorder=10,
                )
                ax.legend(loc="lower right")

        # Plot with the Data Source in the background
        if data_source_for_plt is None:
            data_source_for_plt = self.last_data_source.forecast_data_source

        if t_end is not None:
            t_interval_to_animate = [self.current_data_t_0 + self.reach_times[0], t_end.timestamp()]
        else:
            t_interval_to_animate = [
                self.current_data_t_0 + rel_time
                for rel_time in [self.reach_times[0], self.reach_times[-1]]
            ]

        if with_background:
            data_source_for_plt.animate_data(
                x_interval=[self.grid.domain.lo[0], self.grid.domain.hi[0]],
                y_interval=[self.grid.domain.lo[1], self.grid.domain.hi[1]],
                t_interval=t_interval_to_animate,
                temporal_resolution=temporal_resolution,
                spatial_resolution=spatial_resolution,
                forward_time=forward_time | (self.specific_settings["direction"] == "forward"),
                add_ax_func=add_value_fct_snapshot,
                colorbar=False,
                output=filename,
                fps=fps,
                **background_animation_args,
            )

        else:
            # create global figure object where the animation happens
            if "figsize" in kwargs:
                fig = plt.figure(figsize=kwargs["figsize"])
            else:
                fig = plt.figure(figsize=(12, 12))

            temporal_vector = np.arange(len(self.reach_times))
            if temporal_resolution is not None:
                temporal_vector = np.arange(
                    start=t_interval_to_animate[0],
                    stop=t_interval_to_animate[1],
                    step=temporal_resolution,
                )

            def render_func(time_idx, temporal_vector=temporal_vector):
                # reset plot this is needed for matplotlib.animation
                plt.clf()
                # Step 2: Create ax object
                if data_source_for_plt.source_config_dict["use_geographic_coordinate_system"]:
                    ax = DataSource.set_up_geographic_ax()
                else:
                    ax = plt.axes()
                # get from time_idx to posix_time
                add_value_fct_snapshot(ax, temporal_vector[time_idx])

            # set time direction of the animation
            frames_vector = np.where(
                forward_time,
                np.arange(len(temporal_vector)),
                np.flip(np.arange(len(temporal_vector))),
            )
            # create animation function object (it's not yet executed)
            ani = Animation.FuncAnimation(fig, func=render_func, frames=frames_vector, repeat=False)

            # render the animation with the keyword arguments
            DataSource.render_animation(animation_object=ani, output=filename, fps=fps)
