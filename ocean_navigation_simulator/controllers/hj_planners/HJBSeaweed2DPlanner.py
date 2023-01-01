import math
import os
import pickle
import time
import warnings
from functools import partial
from typing import Dict, Optional, Union

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
import scipy
import xarray as xr

from ocean_navigation_simulator.controllers.hj_planners.HJPlannerBase import (
    HJPlannerBase,
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

        # set first_plan to True so we plan on the first run over the whole time horizon
        self.specific_settings["first_plan"] = True
        self.previous_reach_times, self.previous_all_values = [None] * 2
        #TODO: retrieve forecast_length rather than specifiying it
        if "forecast_length" not in self.specific_settings:
            raise KeyError(
                "\"forecast_length\" is not defined in specific_settings. Please provide the forecast length."
            )

    def get_x_from_full_state(
        self, x: Union[PlatformState, SpatioTemporalPoint, SpatialPoint]
    ) -> jnp.ndarray:
        return jnp.array(x.__array__())[:2]

    def get_time_vector(self, T_max_in_seconds: int) -> int:
        """Return n_time_vector for a given T_max_in_seconds. If we plan over the full horizon we take complete n_time_vector. If we only replan the forecast horizon and take the previous value fct. as initial values we shorten the n_time_vector accordingly"""
        return np.rint((T_max_in_seconds * self.specific_settings["n_time_vector"])/self.specific_settings["T_goal_in_seconds"]).astype(int)

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
            return jnp.zeros(self.nonDimGrid.shape)
        elif direction == "backward":
            return jnp.zeros(self.nonDimGrid.shape)
        elif direction == "multi-time-reach-back":
            raise NotImplementedError(
                "HJPlanner: Multi-Time-Reach not implemented yet"
            )           
        else:
            raise ValueError(
                "Direction in specific_settings of HJPlanner needs to be forward, backward, or multi-reach-back."
            )

    def _get_idx_closest_value_in_array(self, array: np.ndarray, value: Union[int,float]) -> int:
        """Takes a value and an array and returns the index of the closest value in the array.
        Args:
            array: array in which to find the index of the closest value
            value: value for which to find closest array entry / index
        Returns:
            idx: index of closest value in array
        """
        return np.argmin(abs(array - value))

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
            self._extract_trajectory(x_start=self.get_x_from_full_state(x_t))

        elif self.specific_settings["direction"] == "backward":
            # Note: no trajectory is extracted as the value function is used for closed-loop control

            # Check whether we plan the first time over the whole time-horizon or only over days with new forecast and recycle value fct. for the remaining days
            if self.specific_settings["first_plan"]:
                initial_values = self.get_initial_values(direction="backward")
                T_max_in_seconds = self.specific_settings["T_goal_in_seconds"]
            elif not self.specific_settings["first_plan"] and self.specific_settings["forecast_length"] < self.specific_settings["T_goal_in_seconds"]:
                time_idx = self._get_idx_closest_value_in_array(self.previous_reach_times, self.specific_settings["forecast_length"])
                initial_values = self.all_values[time_idx]
                T_max_in_seconds = self.specific_settings["forecast_length"]
                
            self._run_hj_reachability(
                initial_values=initial_values,
                t_start=x_t.date_time,
                T_max_in_seconds=T_max_in_seconds,
                dir="backward",
            )
            
            if self.specific_settings["first_plan"] and self.specific_settings["forecast_length"] < self.specific_settings["T_goal_in_seconds"]:
                # Set first_plan to False after first planning is finished
                self.specific_settings["first_plan"] = False
            elif not self.specific_settings["first_plan"] and self.specific_settings["forecast_length"] < self.specific_settings["T_goal_in_seconds"]:
                # concatenate the the static part and the new part of the value fct. based on new FC data 
                print("concatentate")
                self.reach_times = jnp.append(self.previous_reach_times[:time_idx], self.reach_times, axis=0)
                self.all_values = jnp.append(self.previous_all_values[:time_idx], self.all_values, axis=0)
                

            self._extract_trajectory(x_start=self.get_x_from_full_state(x_t))

            # save current non-flipped reach times and value fct. for next replanning
            self.previous_reach_times = self.reach_times
            self.previous_all_values = self.all_values

            # arrange to forward times by convention for plotting and open-loop control
            self._flip_value_func_to_forward_times()

        elif self.specific_settings["direction"] == "forward-backward":
            raise NotImplementedError(
                "HJPlanner: Forward-Backward not implemented yet"
            )
        elif self.specific_settings["direction"] == "multi-time-reach-back":
            raise NotImplementedError(
                "HJPlanner: Multi-Time-Reach not implemented yet"
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
            x_T=observation.platform_state.to_spatio_temporal_point(),
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
        # path = "./data/seaweed/seaweed_precomputed_over_area.nc"
        # if os.path.exists(path):
        #     seaweed_xarray = xr.open_dataset(path)
        # else:
        seaweed_xarray = self.arena.seaweed_field.hindcast_data_source.get_data_over_area(
            x_interval=x_interval,
            y_interval=y_interval,
            t_interval=t_interval,
            spatial_resolution=self.specific_settings["grid_res"],
        )
        # seaweed_xarray.to_netcdf(path=path)

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

        # import plotly.graph_objects as go
        # import numpy as np

        # # Read data from a csv
        # z = seaweed_xarray["F_NGR_per_second"].data[0]
        # x = self.grid.states[..., 0]
        # y = self.grid.states[..., 1]
        # fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        # # fig.update_layout(title='Mt Bruno Elevation', autosize=False,
        # #                   width=500, height=500,
        # #                   margin=dict(l=65, r=50, b=65, t=90))
        # fig.show()

        # update non_dimensional_dynamics with the new non_dim scaling and offset
        self.nondim_dynamics.characteristic_vec = self.characteristic_vec
        self.nondim_dynamics.offset_vec = self.offset_vec

        self.nondim_dynamics.dimensional_dynamics.characteristic_vec = self.characteristic_vec
        self.nondim_dynamics.dimensional_dynamics.offset_vec = self.offset_vec

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
        with open(folder + "characteristic_vec.pickle", "wb") as file:
            pickle.dump(self.characteristic_vec, file)
        with open(folder + "offset_vec.pickle", "wb") as file:
            pickle.dump(self.characteristic_vec, file)
        with open(folder + "initial_values.pickle", "wb") as file:
            pickle.dump(self.initial_values, file)

    @staticmethod
    def from_saved_planner_state(folder, problem: NavigationProblem, arena: ArenaFactory, verbose: Optional[int] = 0):
        # Settings
        with open(folder + "specific_settings.pickle", "rb") as file:
            specific_settings = pickle.load(file)

        planner = HJBSeaweed2DPlanner(problem=problem, specific_settings=specific_settings, arena=arena)

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