# TEST
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
from scipy.interpolate import interp1d, RectBivariateSpline
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
        c3: Optional[C3Python] = None,
    ):

        # get arena object for accessing seaweed growth model
        self.arena = arena
        self.c3 = c3

        super().__init__(problem, specific_settings)
        self.specific_settings = {
            "platform_dict": problem.platform_dict if problem is not None else None,
            "grid_res": 0.083,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
            "grid_res_average": 0.166,  # Grid res for average data
            "grid_res_seaweed": 0.166,  # Grid res for seaweed data
            "deg_around_xt_xT_box": 1,  # Area over which to run HJ_reachability
            "deg_around_xt_xT_box_average": 10,  # area over which to run HJ_reachability for average data
            "precomputation": False,  # Defines whether value fct. should be precomputed or normal planning is requested
            "value_fct_folder": "temp/precomputed_value_fcts/",  # Where to save and load precomputed value fcts
            "seaweed_precomputation_folder": "ocean_navigation_simulator/package_data/seaweed_growth_maps",
            "take_precomp_seaweed_maps": False,
        } | self.specific_settings
        (
            self.all_values_subset,
            self.all_values_global_flipped,
            self.reach_times_global_flipped_posix,
            self.x_grid_global,
            self.y_grid_global,
            self.seaweed_xarray,
            self.last_observation,
        ) = [None] * 7

        # Set first_plan to True so we plan on the first run over the average data
        self.first_plan = True

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

        # Depending on the reachability direction run the respective algorithm
        if self.specific_settings["direction"] == "forward":
            raise NotImplementedError("HJPlanner: Forward not implemented yet")

        elif self.specific_settings["direction"] == "backward":
            # Note: no trajectory is extracted as the value function is used for closed-loop control

            # Load seaweed data if it doesn't exist yet
            if self.seaweed_xarray is None:
                self._update_seaweed_data()

            # Check whether we plan the first time in order to retrieve the gloabl value fct for the time period after the interval we have FC data available.
            # Only in case FC horizon is shorter than acutal planning horizon --> otherwise we wouldn't need the averages
            if (
                self.first_plan
                and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
            ):

                # Load average data for running HJ
                self._update_average_data()

                # Run data checks if the right current data is loaded in the interpolation function -> With current implementation of _check_data_settings_compatibility() will always raise warning
                # x_t_for_test = x_t
                # x_t_for_test.date_time += timedelta(seconds=self.forecast_length)
                # self._check_data_settings_compatibility(x_t=x_t_for_test)

                # Get inital values and start and T_max for running HJ
                initial_values = self.get_initial_values(direction="backward")
                t_0 = x_t.date_time + timedelta(seconds=self.forecast_length)
                T_max_in_seconds = (
                    self.specific_settings["T_goal_in_seconds"] - self.forecast_length
                )

                self.logger.info("HJBSeaweed2DPlanner: Planning over average data")

                self._run_hj_reachability(
                    initial_values=initial_values,
                    t_start=t_0,
                    T_max_in_seconds=T_max_in_seconds,
                    dir="backward",
                )

                # Saving global values and reach times for later use
                self.all_values_global = self.all_values
                self.reach_times_global = self.reach_times
                self.reach_times_global_posix = self.reach_times_global + self.current_data_t_0

                self.first_plan = False

                # Load forecast data for running consecutive HJ run
                self._update_forecast_data()

                # Run data checks if the right current data is loaded in the interpolation function -> With current implementation of _check_data_settings_compatibility() will always raise warning
                # self._check_data_settings_compatibility(x_t=x_t)

                # Get initial values for backward computation over FC horizon
                initial_values = self._get_ininital_values_from_global_values()
                T_max_in_seconds = self.forecast_length

            # If we have already computed our average value fct.
            elif (
                not self.first_plan
                and self.forecast_length < self.specific_settings["T_goal_in_seconds"]
            ):
                # Load forecast data for running consecutive HJ
                self._update_forecast_data()

                # Run data checks if the right current data is loaded in the interpolation function -> With current implementation of _check_data_settings_compatibility() will always raise warning
                # self._check_data_settings_compatibility(x_t=x_t)

                # Get initial values for backward computation over FC horizon
                initial_values = self._get_ininital_values_from_global_values()
                T_max_in_seconds = self.forecast_length

            # Check if planning horizon is actual shorter than FC horizon - no need for averages
            elif self.forecast_length >= self.specific_settings["T_goal_in_seconds"]:
                # Load forecast data for running consecutive HJ
                self._update_forecast_data()

                # Run data checks if the right current data is loaded in the interpolation function
                self._check_data_settings_compatibility(x_t=x_t)

                # Get inital values and T_max for running HJ
                initial_values = self.get_initial_values(direction="backward")
                T_max_in_seconds = self.specific_settings["T_goal_in_seconds"]

            self.logger.info("HJBSeaweed2DPlanner: Planning over forecast data")
            # Run HJ with previously set configuration
            self._run_hj_reachability(
                initial_values=initial_values,
                t_start=x_t.date_time,
                T_max_in_seconds=T_max_in_seconds,
                dir="backward",
            )

            # FOR DEBUG: Analyze how often we get issues with propagating boundry conditions
            # Get indices of x_t to access mask on relevant position
            lon_idx = self._get_idx_closest_value_in_array(self.grid.states[:, 0, 0], x_t.lon.deg)
            lat_idx = self._get_idx_closest_value_in_array(self.grid.states[0, :, 1], x_t.lat.deg)

            # arrange to forward times by convention for plotting and open-loop control
            self._set_value_func_to_forward_time()

            # Check whether the value of the first timestep on position x_t is higher than 0.8 which we estimate to be from propagated boundry conditions
            if self.all_values[0][lon_idx][lat_idx] > 0.85:
                # raise ValueError
                print(
                    f"Value function has invalid value {self.all_values[0][lon_idx][lat_idx]} at x_t{x_t}, due to propagating boundry conditions."
                )
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
           Will be mainly passed due to alternative implementations in this class:
           update_average_data(), update_forecast_data() and _update_seaweed_data()
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        """
        # Save because we are using two other function update_average_data() and update_forecast_data() which otherwise don't have access to the observation
        # We do this because we have to use different data in the _plan function.
        self.last_observation = observation

        # Set forecast length
        # TODO: check if forecast_data_source.forecast_data_source still necessary - probably not!
        # Extract FC length in seconds -> if else in order to also work with toy examples i.e current highway
        # TODO: change to posix time

        # Check if FC source is from type ForecastFromHindcastSource and retrieve forecast length from source config
        if isinstance(
            observation.forecast_data_source.forecast_data_source,
            ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource.ForecastFromHindcastSource,
        ):
            self.forecast_length = (
                observation.forecast_data_source.forecast_data_source.forecast_length_in_days
                * 3600
                * 24
            )
        # Check if FC source is real FC source and not from type HindcastFileSource and take maximum available data as forecast_length
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
        # If FC source is just taken from the HC (type HindcastFileSource & therefore HC only setting) take default value of T_goal_in_seconds + buffer 1 day as forecast_length
        else:
            self.forecast_length = self.specific_settings["T_goal_in_seconds"] + 1 * 24 * 3600

        # Set our T_FC which defines the end of our FC horizon in POSIX
        self.forecast_end_time_posix = (
            observation.platform_state.date_time.timestamp() + self.forecast_length
        )

    def _update_seaweed_data(self):
        """Helper function to prepare seaweed data."""
        start = time.time()
        observation = self.last_observation

        # Get data over complete planning horizon so we can reuse the seaweed source and only have to compute it once
        # Step 1: get the x,y,t bounds for current position, goal position and settings.
        t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
            x_0=observation.platform_state.to_spatio_temporal_point(),
            x_T=observation.platform_state.to_spatio_temporal_point(),
            deg_around_x0_xT_box=self.specific_settings["deg_around_xt_xT_box_average"],
            temp_horizon_in_s=self.specific_settings["T_goal_in_seconds"],
        )

        # adjust if specified explicitly in settings
        if "x_interval_seaweed" in self.specific_settings:
            x_interval = self.specific_settings["x_interval_seaweed"]
        if "y_interval_seaweed" in self.specific_settings:
            y_interval = self.specific_settings["y_interval_seaweed"]

        if self.specific_settings["take_precomp_seaweed_maps"]:
            files_dicts = self._get_file_dicts(
                folder=self.specific_settings["seaweed_precomputation_folder"]
            )

            seaweed_xarray = xr.concat(
                [xr.open_dataset(h_dict["file"]) for h_dict in files_dicts], dim="time"
            )
            # TODO: enforce timezone awareness to mitigate warning: Indexing a timezone-naive DatetimeIndex with a timezone-aware datetime is deprecated and will raise KeyError in a future version.  Use a timezone-naive object instead.
            self.seaweed_xarray = seaweed_xarray.sel(time=slice(t_interval[0], t_interval[1]))
        else:
            seaweed_xarray = self.arena.seaweed_field.hindcast_data_source.get_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                t_interval=t_interval,
                spatial_resolution=self.specific_settings["grid_res_seaweed"],
            )

            self.seaweed_xarray = seaweed_xarray.assign(
                relative_time=lambda x: units.get_posix_time_from_np64(x.time)
                - units.get_posix_time_from_np64(seaweed_xarray["time"][0])
            )

            # self.seaweed_xarray.to_netcdf(
            #     f'/Users/matthiaskiller/Desktop/data/seaweed_growth_maps/growth_map_{t_interval[0].strftime("%Y-%m-%dT%H-%M-%SZ")}_{t_interval[1].strftime("%Y-%m-%dT%H-%M-%SZ")}'
            # )

            # raise ValueError("seaweed growth map precomp. done")

        self.logger.info(f"HJBSeaweed2DPlanner: Loading Seaweed Data ({time.time() - start:.1f}s)")

    def _update_average_data(self):
        """Helper function to load the average current data into the interpolation."""
        start = time.time()
        observation = self.last_observation

        temp_horizon_in_s = self.specific_settings["T_goal_in_seconds"] - self.forecast_length
        x_0 = observation.platform_state.to_spatio_temporal_point()
        x_0.date_time += timedelta(seconds=self.forecast_length)

        # Step 1: get the x,y,t bounds for current position, goal position and settings.
        t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
            x_0=x_0,
            x_T=observation.platform_state.to_spatio_temporal_point(),
            deg_around_x0_xT_box=self.specific_settings["deg_around_xt_xT_box_average"],
            temp_horizon_in_s=temp_horizon_in_s,
        )
        # Adjust if specified explicitly in settings
        if "x_interval" in self.specific_settings:
            x_interval = self.specific_settings["x_interval"]
        if "y_interval" in self.specific_settings:
            y_interval = self.specific_settings["y_interval"]

        # Get the average data subset from the file
        data_xarray = observation.average_data_source.get_data_over_area(
            x_interval=x_interval,
            y_interval=y_interval,
            t_interval=t_interval,
            temporal_resolution=7200,  # TODO: add configurable temporal_resolution
            spatial_resolution=self.specific_settings["grid_res_average"],
        )

        # Reduce temporal margins since the averages will be returned with large temporal margins (+- 1 month)
        dt = data_xarray["time"][1] - data_xarray["time"][0]
        data_xarray = data_xarray.sel(
            time=slice(
                np.datetime64(t_interval[0].replace(tzinfo=None)) - dt,
                np.datetime64(t_interval[1].replace(tzinfo=None)) + dt,
            )
        )

        # Calculate relative posix_time (we use it in interpolation because jax uses float32 and otherwise cuts off)
        data_xarray = data_xarray.assign(
            relative_time=lambda x: units.get_posix_time_from_np64(x.time)
            - units.get_posix_time_from_np64(data_xarray["time"][0])
        )

        # Feed in the current data to the Platform classes
        self.dim_dynamics.update_jax_interpolant(data_xarray, self.seaweed_xarray)

        # Set absolute time in UTC Posix time
        self.current_data_t_0 = units.get_posix_time_from_np64(data_xarray["time"][0]).data
        # Set absolute final time in UTC Posix time
        self.current_data_t_T = units.get_posix_time_from_np64(data_xarray["time"][-1]).data

        # Initialize the grids and dynamics to solve the PDE with
        self.initialize_hj_grid(data_xarray)

        # Save global grid for late use
        self.x_grid_global = self.grid.states[:, 0, 0]
        self.y_grid_global = self.grid.states[0, :, 1]

        # Delete the old caches (might not be necessary for analytical fields -> investigate)
        self.logger.debug("HJBSeaweed2DPlanner: Cache Size " + str(hj.solver._solve._cache_size()))
        hj.solver._solve._clear_cache()

        self.logger.info(
            f"HJBSeaweed2DPlanner: Loading Average Current Data ({time.time() - start:.1f}s)"
        )

    def _update_forecast_data(self):
        """Helper function to load the forecast current data into the interpolation."""
        start = time.time()
        observation = self.last_observation

        # Step 1: get the x,y,t bounds for current position, goal position and settings.
        t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
            x_0=observation.platform_state.to_spatio_temporal_point(),
            x_T=observation.platform_state.to_spatio_temporal_point(),
            deg_around_x0_xT_box=self.specific_settings["deg_around_xt_xT_box"],
            temp_horizon_in_s=self.forecast_length,
        )
        # adjust if specified explicitly in settings
        if "x_interval" in self.specific_settings:
            x_interval = self.specific_settings["x_interval"]
        if "y_interval" in self.specific_settings:
            y_interval = self.specific_settings["y_interval"]

        # Take forecast data if we do not precompute the value function
        data_xarray = observation.forecast_data_source.get_data_over_area(
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

        # feed in the current data to the Platform classes
        self.dim_dynamics.update_jax_interpolant(data_xarray, self.seaweed_xarray)

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

        self.logger.info(
            f"HJBSeaweed2DPlanner: Loading new Forecast Current Data ({time.time() - start:.1f}s)"
        )

    def _get_file_dicts(self, folder: AnyStr) -> List[dict]:
        """
        Creates an list of dicts ordered according to time available, one for each nc file available in folder.
        The dicts for each file contains:
        {'t_range': [<datetime object>, T], 'file': <filepath> ,'y_range': [min_lat, max_lat], 'x_range': [min_lon, max_lon]}
        """
        # Step 1: get a list of files from the folder
        files_list = []

        # Allow for list of files/folders or mixture as input
        # This is useful to prevent loading hundreds of files!
        # Fully backwards compatible
        if type(folder) is not list:
            folder = [folder]

        for place in folder:
            if os.path.isdir(place):
                new_files = [
                    place + f
                    for f in os.listdir(place)
                    if (os.path.isfile(os.path.join(place, f)) and f != ".DS_Store")
                ]
                files_list += new_files
            elif os.path.isfile(place):
                files_list.append(place)

        # Step 2: iterate over all files to extract the grids and put them in an ordered list of dicts
        list_of_dicts = []
        for file in files_list:
            grid_dict = self._get_grid_dict_from_file(file)
            # append the file to it:
            grid_dict["file"] = file
            list_of_dicts.append(grid_dict)

        # Step 3: sort the list
        list_of_dicts.sort(key=lambda dict: dict["t_range"][0])

        return list_of_dicts

    def _get_grid_dict_from_file(self, file: AnyStr) -> dict:
        """Helper function to create a grid dict from a local nc3 file."""
        f = xr.open_dataset(file)
        # get the time coverage in POSIX
        t_grid = units.get_posix_time_from_np64(f.variables["time"].data)
        y_range = [f.variables["lat"].data[0], f.variables["lat"].data[-1]]
        x_range = [f.variables["lon"].data[0], f.variables["lon"].data[-1]]
        # close netCDF file
        f.close()
        # create dict
        return {
            "t_range": [
                datetime.fromtimestamp(t_grid[0], timezone.utc),
                datetime.fromtimestamp(t_grid[-1], timezone.utc),
            ],
            "y_range": y_range,
            "x_range": x_range,
        }

    def _get_ininital_values_from_global_values(self):
        """Helper function to retrieve the initial values for backward computation over FC horizon for a given forecast end time (temporal) and on self.grid (spatial).
            Note: Must be called after calling the _update_forecast_data() function.
        Returns:
            values: interpolated value subset corresponding to the given posix time and self.grid
        """
        # Step 1: Get temporally interpolated slice of global value fct. at forecast_end_time_posix
        initial_values = interp1d(
            self.reach_times_global_posix,
            self.all_values_global,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )(self.forecast_end_time_posix).squeeze()

        # Step 2: Get spatially interpolated subset of slice
        interpolator = RectBivariateSpline(self.x_grid_global, self.y_grid_global, initial_values)
        initial_values = interpolator(
            self.grid.coordinate_vectors[0], self.grid.coordinate_vectors[1]
        )

        return initial_values

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
