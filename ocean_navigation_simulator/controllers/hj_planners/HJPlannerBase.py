import abc
import math
import os
import pickle
import time
from bisect import bisect
from datetime import datetime, timezone
from functools import partial
from typing import AnyStr, Callable, List, Optional, Tuple, Union

# Note: if you develop on hj_reachability repo and this library simultaneously, add the local version with this line
# sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) + 'hj_reachability_c3'])
import hj_reachability as hj
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr
from scipy.interpolate import interp1d

from ocean_navigation_simulator.controllers.Controller import Controller
from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.utils import units


class HJPlannerBase(Controller):
    """
    Baseclass for all HJ reachability-based Planners using backwards/forwards/multi-time reachability.
        For details see: "A future for intelligent autonomous ocean observing systems" P.F.J. Lermusiaux

        Note: The Baseclass is general and works for 2D, 3D, 4D System.
        In the Baseclass, the PDE is solved in non_dimensional dynamics in space and time to reduce numerical errors.
        To use this class, only the 'abstractmethod' functions need to be implemented.

        See Planner class for the rest of the attributes.
    """

    def __init__(self, problem: NavigationProblem, specific_settings: dict):
        """
        Constructor for the HJ Planner Baseclass.
        Args:
            problem: the Problem the controller will run on
            specific_settings: Attributes required in the specific_settings dict
                direction: string of {'forward', 'backward', 'forward-backward', 'multi-time-reach-back'}
                    Which directional setting for the reachability to run.
                n_time_vector: int
                    The number of elements in the time vector which determines how granular the value function is saved.
                deg_around_xt_xT_box: float
                    how much degree lon, lat around the x_t, x_T we cut out to plan on.
                accuracy: string one of {"low", "medium", "high", "very_high"}
                    Determines the accuracy with which the PDE solver is run.
                artificial_dissipation_scheme: string one of {"global", "local", "local_local"}
                    Determines which dissipation term is used in the Lax-Friedrichs Time-integration approximation.
                T_goal_in_h: float
                    If backward: the final time to be used for back propagating the reachable set.
                    If forward:  the time until which the forward reachable set is propagated.
                initial_set_radii: list of floats e.g.  [0.03, 0.03] for 2D
                    The radii of the ellipsoidal initial set of the value function in degree the respective dimensions.
                grid_res: tuple (int, int) for 2D, (int, int, int) for 3D Grid etc.
                    The granularity of the grid on which the HJ Reachability is computed.
                use_geographic_coordinate_system:
                    If True we use the Geographic coordinate system in lat, lon degree (divide by fixed amount to convert)
                    If False then the coordinate system and speeds of the agent are in m/s.
        """
        super().__init__(problem)
        self.specific_settings = {
            "replan_on_new_fmrc": True,
            "replan_every_X_seconds": None,
            "n_time_vector": 199,
            "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
            "accuracy": "high",
            "artificial_dissipation_scheme": "local_local",
            "use_geographic_coordinate_system": True,
            "progress_bar": False,
            # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
            "d_max": 0.0,
            "platform_dict": problem.platform_dict,
        } | specific_settings

        # initialize vectors for open_loop control
        self.times, self.x_traj, self.contr_seq = None, None, None

        # saving the planned trajectories for inspection purposes
        self.planned_trajs = []

        # create a variable that persists across runs of self.plan() to reference the currently reload data
        self.current_data_t_0, self.current_data_t_T = [None] * 2
        # Two variables to enable both re-planning with fixed frequency and only when new forecast available
        self.last_data_source, self.last_fmrc_time_planned_with, self.last_planning_posix = [
            None
        ] * 3

        # this is just a variable that persists after planning for plotting/debugging
        self.x_t = None

        # Initializes variables needed when solving the HJ PDE, they will be filled in the plan method
        self.reach_times, self.all_values, self.grid = [None] * 3
        self.diss_scheme = self._get_dissipation_schema()
        self.x_traj, self.contr_seq, self.distr_seq = [None] * 3
        # Initialize variables needed for solving the PDE in non_dimensional terms
        self.characteristic_vec, self.offset_vec, self.nonDimGrid, self.nondim_dynamics = [None] * 4

        self.planner_cache_index = 0

        # Initialize the non_dimensional_dynamics and within it the dimensional_dynamics
        # Note: as initialized here, it's not usable, only after 'update_current_data' is called for the first time.
        self.nondim_dynamics = hj.dynamics.NonDimDynamics(
            dimensional_dynamics=self.get_dim_dynamical_system()
        )

        if (
            self.specific_settings["d_max"] > 0
            and self.specific_settings["direction"] == "multi-time-reach-back"
        ):
            self.logger.info(
                "No disturbance implemented for multi-time reachability, only runs with d_max=0."
            )

    def get_open_loop_control_from_plan(self, state: PlatformState) -> PlatformAction:
        """Indexing into the planned open_loop control sequence using the time from state.
        Args:
            state    PlatformState containing [lat, lon, battery_level, date_time]
        Returns:
            PlatformAction object
        """
        # an easy way of finding for each time, which index of control signal to apply
        idx = bisect(self.times, state.date_time.timestamp()) - 1
        if idx == len(self.times) - 1:
            idx = idx - 1
            self.logger.warning(
                "Controller Warning: continuing using last control although not planned as such"
            )

        # extract right element from ctrl vector
        return PlatformAction(magnitude=self.contr_seq[0, idx], direction=self.contr_seq[1, idx])

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """Main interface function for the simulation, so all the logic to trigger re-planning is inside here.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        Returns:
            PlatformAction dataclass
        """
        # Step 1: Check if we should re-plan based on specified criteria
        self.replan_if_necessary(observation)

        # Step 2: return the action from the plan
        start = time.time()
        action = self._get_action_from_plan(state=observation.platform_state)
        self.logger.debug(f"HJPlannerBase: Get Action from Plan ({time.time() - start:.1f}s)")
        return action

    def get_action_over_horizon(
        self, observation: ArenaObservation, horizon: int, dt_in_sec: int
    ) -> List[PlatformAction]:

        # Step 1: Check if we should re-plan based on specified criteria
        self.replan_if_necessary(observation)
        t_start = observation.platform_state.date_time.timestamp()
        x_start = self.get_x_from_full_state(observation.platform_state)
        dt_in_sec = self.specific_settings["platform_dict"]["dt_in_s"]
        # Step 2: return the action from the plan
        backtracking_reach_times, backtracking_all_values = [
            np.flip(seq, axis=0) for seq in [self.reach_times, self.all_values]
        ]
        t_rel_stop = t_start - self.current_data_t_0 + self.reach_times[0]
        t_rel_start = self.reach_times[-1]
        if t_rel_start <= t_rel_stop:
            raise ValueError("t_start is after the last time a value function is available.")

        traj_rel_times_vector = np.arange(
            start=t_rel_start,
            stop=t_rel_stop,
            step=dt_in_sec if self.specific_settings["direction"] == "forward" else -dt_in_sec,
        )

        shortend = traj_rel_times_vector[-horizon:]
        start = time.time()
        (
            times,
            x_traj,
            contr_seq,
            distr_seq,
        ) = self.nondim_dynamics.dimensional_dynamics.backtrack_trajectory(
            grid=self.grid,
            x_init=x_start,
            times=backtracking_reach_times,
            all_values=backtracking_all_values,
            traj_times=shortend,
        )
        self.logger.debug(f"HJPlannerBase: Get Action from Plan ({time.time() - start:.1f}s)")
        x_traj, contr_seq, distr_seq = [
            np.flip(seq, axis=1) for seq in [x_traj, contr_seq, distr_seq]
        ]
        return [
            PlatformAction(magnitude=contr_seq[0, idx], direction=contr_seq[1, idx])
            for idx in range(horizon - 1)
        ]

    def replan_if_necessary(self, observation: ArenaObservation) -> bool:
        if self._check_for_replanning(observation):
            if "load_plan" in self.specific_settings and self.specific_settings["load_plan"]:
                folder = f'{self.specific_settings["planner_path"]}forecast_planner_idx_{self.planner_cache_index+1}/'
                self.restore_state(folder=folder)
            else:
                start = time.time()

                # log x_t and data_source for plotting and easier access later
                self.x_t = observation.platform_state
                self.last_data_source = observation.forecast_data_source.forecast_data_source
                # Update the data used in the HJ Reachability Planning
                self._update_current_data(observation=observation)
                self._plan(observation.platform_state)
                self.set_interpolator()

                if (
                    "save_after_planning" in self.specific_settings
                    and self.specific_settings["save_after_planning"]
                ):
                    self.save_planner_state(
                        f'{self.specific_settings["planner_path"]}forecast_planner_idx_{self.planner_cache_index}/'
                    )
                    self.planner_cache_index += 1

                self.logger.info(
                    f"HJPlannerBase: Re-planning finished ({time.time() - start:.1f}s)"
                )
            return True
        return False

    def _check_for_replanning(self, observation: ArenaObservation) -> bool:
        """Helper Function to check if we want to replan with HJ Reachability.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        """

        # Check for re-planning with new forecast (or first time so no idx set yet)
        if (
            self.specific_settings.get("replan_on_new_fmrc", False)
            or self.last_fmrc_time_planned_with is None
        ):
            old = self.last_fmrc_time_planned_with
            if self._new_forecast_data_available(observation):
                # If the data_source is an observer, delete all error measurements from the old forecast.
                if isinstance(self.last_data_source, Observer):
                    self.last_data_source.reset()
                self.logger.info(
                    f"HJPlannerBase: Planning because of new forecast (Old: {old}, New: {self.last_fmrc_time_planned_with})"
                )
                return True

        # Check for re-planning after fixed time intervals
        if self.specific_settings["replan_every_X_seconds"]:
            if (
                observation.platform_state.date_time.timestamp() - self.last_planning_posix
                >= self.specific_settings["replan_every_X_seconds"]
            ):
                self.logger.info("HJPlannerBase: Planning because of fixed time interval.")
                return True
        return False

    def _new_forecast_data_available(self, observation: ArenaObservation) -> bool:
        """Helper function to check if new forecast data is available in the forecast_data_source.
        If yes, the data in the interpolation is updated and True is returned, otherwise False.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        """
        # Get the idx for the most recent file
        most_current_fmrc_time = (
            observation.forecast_data_source.check_for_most_recent_fmrc_dataframe(
                time=observation.platform_state.date_time
            )
        )
        # Check if this is after our last planned one
        if most_current_fmrc_time != self.last_fmrc_time_planned_with:
            # update the current data in the jax interpolatn
            self.last_fmrc_time_planned_with = most_current_fmrc_time
            return True
        else:
            return False

    def _get_action_from_plan(self, state: PlatformState) -> PlatformAction:
        """Extracts the next action from the most recent plan (saved value function).
        if specific_settings['direction'] is forward:     applying the actions open-loop from the contr_seq
        if specific_settings['direction'] is backward:    computing the gradient/action directly from the value function

        Args:
            state: PlatformState containing the location & time
        Returns:
            PlatformAction to send to the simulation
        """
        if self.specific_settings["direction"] == "forward" or not self.specific_settings.get(
            "closed_loop", True
        ):
            u_out = self.get_open_loop_control_from_plan(state=state)
        else:
            # check if time is outside times and throw warning if yes but continue with existing value function.
            rel_time = state.date_time.timestamp() - self.current_data_t_0
            if rel_time > self.reach_times[-1]:
                self.logger.warning(
                    "HJPlanner Warning: Extrapolating time beyond the reach_times, should replan."
                )
                rel_time = self.reach_times[-1]

            # Extract the optimal control from the calculated value function at the specific platform state.
            try:
                u_out, _ = self.nondim_dynamics.dimensional_dynamics.get_opt_ctrl_from_values(
                    grid=self.grid,
                    x=self.get_x_from_full_state(state),
                    time=rel_time,
                    times=self.reach_times,
                    all_values=self.all_values,
                )
            except BaseException:
                print(f"self.last_fmrc_time_planned_with: {self.last_fmrc_time_planned_with}")
                print(f"rel_time: {rel_time:.0f}")
                print(f"self.reach_times: [{self.reach_times[0]:.0f}, {self.reach_times[-1]:.0f}]")
                raise

        return PlatformAction(magnitude=u_out[0], direction=u_out[1])

    def get_waypoints(self) -> List[SpatioTemporalPoint]:
        """Returns: a list of waypoints each containing [lon, lat, time]"""
        return np.vstack((self.x_traj, self.times)).T.tolist()

    def _get_dissipation_schema(self):
        """Helper function to directly set the artificial dissipation schema used in solving the PDE."""
        # Note: can be done more elegantly by list_indexing =)
        if self.specific_settings["artificial_dissipation_scheme"] == "local_local":
            return hj.artificial_dissipation.local_local_lax_friedrichs
        elif self.specific_settings["artificial_dissipation_scheme"] == "local":
            return hj.artificial_dissipation.local_lax_friedrichs
        elif self.specific_settings["artificial_dissipation_scheme"] == "global":
            return hj.artificial_dissipation.global_lax_friedrichs
        else:
            raise ValueError(
                "artificial_dissipation_scheme is not one of {global, local, local_local}"
            )

    @abc.abstractmethod
    def initialize_hj_grid(self, xarray: xr):
        """Initialize grid to solve PDE on."""
        pass

    @abc.abstractmethod
    def get_dim_dynamical_system(self) -> hj.dynamics.Dynamics:
        """Creates the dimensional dynamics object and returns it."""
        pass

    @abc.abstractmethod
    def get_initial_values(self, direction: AnyStr) -> jnp.ndarray:
        """Create the initial value function over the grid must be implemented by specific planner."""
        pass

    @abc.abstractmethod
    def get_x_from_full_state(self, x: Union[SpatialPoint, PlatformState, SpatioTemporalPoint]):
        """Return the x state appropriate for the specific reachability planner."""
        pass

    def _check_data_settings_compatibility(self, x_t: PlatformState):
        """Helper function to check data availability before running HJ Reachability to prevent errors later."""
        # Check if x_t is in the forecast times and transform to rel_time in seconds
        if x_t.date_time.timestamp() < self.current_data_t_0:
            error_string = (
                "Current time {} is before the start of the forecast data. This should not happen. "
                "current_data_t_0 is {}".format(
                    x_t.date_time, datetime.fromtimestamp(self.current_data_t_0, tz=timezone.utc)
                )
            )
            self.logger.error(error_string)
            raise ValueError(error_string)
        # Check if the current_data is sufficient for planning over the specified time horizon.
        # if not enough we throw a warning
        if (
            x_t.date_time.timestamp() + self.specific_settings["T_goal_in_seconds"]
            > self.current_data_t_T
        ):
            self.logger.warning(
                "Loaded forecast data does not contain the full time-horizon from x_t {} to T_goal_in_seconds {}. "
                "Automatically adjusting to maximum available time in this forecast.".format(
                    x_t.date_time, self.specific_settings["T_goal_in_seconds"]
                )
            )

    def _plan(self, x_t: PlatformState):
        """Main function where the reachable front is computed.
        Args:
            x_t: Platform state used as start/target of HJ Reachability computation, depending on 'direction'.
        """
        # self.problem.start_state = x_t
        # run data checks if the right current data is loaded in the interpolation function
        self._check_data_settings_compatibility(x_t=x_t)

        # Step 2: depending on the reachability direction run the respective algorithm
        if self.specific_settings["direction"] == "forward":
            self._run_hj_reachability(
                initial_values=self.get_initial_values(direction="forward"),
                t_start=x_t.date_time,
                T_max_in_seconds=self.specific_settings["T_goal_in_seconds"],
                dir="forward",
                x_reach_end=self.get_x_from_full_state(self.problem.end_region),
            )
            # log values for closed-loop trajectory extraction
            x_start_backtracking = self.get_x_from_full_state(self.problem.end_region)
            t_start_backtracking = (
                x_t.date_time.timestamp() + self.specific_settings["T_goal_in_seconds"]
            )

        elif self.specific_settings["direction"] == "backward":
            # Note: no trajectory is extracted as the value function is used for closed-loop control
            self._run_hj_reachability(
                initial_values=self.get_initial_values(direction="backward"),
                t_start=x_t.date_time,
                T_max_in_seconds=self.specific_settings["T_goal_in_seconds"],
                dir="backward",
            )
            # arrange to forward times by convention for plotting and open-loop control
            self._set_value_func_to_forward_time()
            # log values for closed-loop trajectory extraction
            x_start_backtracking = self.get_x_from_full_state(x_t)
            t_start_backtracking = x_t.date_time.timestamp()

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
            # arrange to forward times by convention for plotting and open-loop control
            self._set_value_func_to_forward_time()
            # log values for closed-loop trajectory extraction
            x_start_backtracking = self.get_x_from_full_state(x_t)
            t_start_backtracking = x_t.date_time.timestamp()
        elif self.specific_settings["direction"] == "multi-time-reach-back":
            # Step 1: run multi-reachability backwards in time
            self._run_hj_reachability(
                initial_values=self.get_initial_values(direction="multi-time-reach-back"),
                t_start=x_t.date_time,
                T_max_in_seconds=self.specific_settings["T_goal_in_seconds"],
                dir="multi-time-reach-back",
            )
            # arrange to forward times by convention for plotting and open-loop control
            self._set_value_func_to_forward_time()

            # log values for closed-loop trajectory extraction
            x_start_backtracking = self.get_x_from_full_state(x_t)
            t_start_backtracking = x_t.date_time.timestamp()
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

    def _run_hj_reachability(
        self,
        initial_values: jnp.ndarray,
        t_start: datetime,
        T_max_in_seconds: float,
        dir: AnyStr,
        x_reach_end: jnp.ndarray = None,
    ):
        """Run hj reachability starting with initial_values at t_start for maximum of T_max_in_seconds
        or until x_reach_end is reached going in the time direction of dir.

        Args:
        initial_values:    value function of the initial set, must be same dim as grid.ndim
        t_start:           starting datetime object (absolute time, not relative time yet)
        T_max_in_seconds:  maximum time to run forward reachability for in seconds
        dir:               direction for reachability either 'forward' or 'backward'
        x_reach_end:       Optional: target point, must be same dim as grid.ndim (Later can be a region)
                            stopping the front computation when the target state is reached.

        Output:             None, everything is set as class variable
        """

        # set the time_scales and offset in the non_dim_dynamics in which the PDE is solved
        self.nondim_dynamics.tau_c = min(
            T_max_in_seconds, int(self.current_data_t_T - t_start.timestamp())
        )
        # This is in relative seconds since current data t_0
        self.nondim_dynamics.t_0 = t_start.timestamp() - self.current_data_t_0

        # save initial_values for later access
        self.initial_values = initial_values

        # set up the non_dimensional time-vector for which to save the value function
        solve_times = np.linspace(0, 1, self.specific_settings["n_time_vector"] + 1)

        self.logger.info(f"HJPlannerBase: Running {dir}")

        if dir == "backward" or dir == "multi-time-reach-back":
            solve_times = np.flip(solve_times, axis=0)
            self.nondim_dynamics.dimensional_dynamics.control_mode = "min"
            self.nondim_dynamics.dimensional_dynamics.disturbance_mode = "max"
        elif dir == "forward":
            self.nondim_dynamics.dimensional_dynamics.control_mode = "max"
            self.nondim_dynamics.dimensional_dynamics.disturbance_mode = "min"

        # specific settings for multi-time-reach-back
        if dir == "multi-time-reach-back":
            # write multi_reach hamiltonian postprocessor
            def multi_reach_step(mask, val):
                val = jnp.where(mask <= 0, -1, val)
                return val

            # combine it with partial sp the mask input gets fixed and only val is open
            p_multi_reach_step = partial(multi_reach_step, initial_values)
            # set the postprocessor to be fed into solver_settings
            hamiltonian_postprocessor = p_multi_reach_step
        else:  # make the postprocessor the identity

            def hamiltonian_postprocessor(value):
                return value

        # create solver settings object
        solver_settings = hj.SolverSettings.with_accuracy(
            accuracy=self.specific_settings["accuracy"],
            x_init=self._get_non_dim_state(x_reach_end) if x_reach_end is not None else None,
            artificial_dissipation_scheme=self.diss_scheme,
            hamiltonian_postprocessor=hamiltonian_postprocessor,
        )

        # solve the PDE in non_dimensional to get the value function V(s,t)
        start = time.time()
        non_dim_reach_times, self.all_values = hj.solve(
            solver_settings=solver_settings,
            dynamics=self.nondim_dynamics,
            grid=self.nonDimGrid,
            times=solve_times,
            initial_values=initial_values,
            progress_bar=self.specific_settings["progress_bar"],
        )
        self.logger.info(f"HJPlannerBase: hj.solve finished ({time.time() - start:.1f}s)")

        # scale up the reach_times to be dimensional_times in seconds again
        self.reach_times = (
            non_dim_reach_times * self.nondim_dynamics.tau_c + self.nondim_dynamics.t_0
        )

    def _get_t_earliest_for_target_region(self) -> Tuple:
        """Helper Function to get the earliest time the forward reachable set overlaps with the target region."""
        # get target_region_mask
        target_region_mask = self.get_initial_values(direction="backward") <= 0

        # iterate forward to get earliest time it's inside
        reached = False
        for idx in range(self.all_values.shape[0]):
            reached = np.logical_and(target_region_mask, self.all_values[idx, ...] <= 0).any()
            if reached:
                break
        # extract earliest relative time of idx
        T_earliest_in_seconds = self.reach_times[idx] - self.reach_times[0]
        if not reached:
            self.logger.info("Not reached, returning maximum time for the backwards reachability.")
        return reached, T_earliest_in_seconds

    def _extract_trajectory(
        self,
        x_start: jnp.ndarray,
        t_start: float,  # in posix time!
        num_traj_disc: Optional[int] = None,
        dt_in_sec: Optional[int] = None,
    ):
        """Backtrack the reachable front to extract a trajectory etc.

        Args:
        x_start:                   start_point for backtracking must be same dim as grid.ndim
        traj_rel_times_vector:     the times vector for which to extract trajectory points for
                                    in seconds from the start of the reachability computation t=0.
                                    Defaults to self.reach_times.
        termination_condn:          function to evaluate if the calculation should be terminated (e.g. because reached)
        """
        # Note: the value function self.all_values and self.reach_times are per default in forward time!

        # Step 0: get all_values and reach_times in the correct direction
        if self.specific_settings["direction"] == "forward":
            backtracking_reach_times, backtracking_all_values = self.reach_times, self.all_values
            t_rel_start = t_start - self.current_data_t_0
            t_rel_stop = self.reach_times[-1]
            if t_rel_start >= t_rel_stop:
                raise ValueError("t_start is after the last time a value function is available.")
        else:
            backtracking_reach_times, backtracking_all_values = [
                np.flip(seq, axis=0) for seq in [self.reach_times, self.all_values]
            ]
            t_rel_stop = t_start - self.current_data_t_0 + self.reach_times[0]
            t_rel_start = self.reach_times[-1]
            if t_rel_start <= t_rel_stop:
                raise ValueError("t_start is after the last time a value function is available.")

        # Step 1: get the traj_rel_times_vector
        if num_traj_disc:
            traj_rel_times_vector = np.linspace(
                start=t_rel_start, stop=t_rel_stop, num=num_traj_disc, endpoint=True
            )
        elif dt_in_sec:
            traj_rel_times_vector = np.arange(
                start=t_rel_start,
                stop=t_rel_stop,
                step=dt_in_sec if self.specific_settings["direction"] == "forward" else -dt_in_sec,
            )
        # setting default times vector for the trajectory
        else:  # default is the same as
            traj_rel_times_vector = backtracking_reach_times

        # Set up termination condition (only needed for multi-time backwards reachability)
        def termination_condn(x_target, r, x, t):
            return np.linalg.norm(x_target - x) <= r

        termination_condn = partial(
            termination_condn,
            jnp.array(self.problem.end_region.__array__()),
            self.problem.target_radius,
        )

        # Step 2: backtrack to get the trajectory
        (
            times,
            x_traj,
            contr_seq,
            distr_seq,
        ) = self.nondim_dynamics.dimensional_dynamics.backtrack_trajectory(
            grid=self.grid,
            x_init=x_start,
            times=backtracking_reach_times,
            all_values=backtracking_all_values,
            traj_times=traj_rel_times_vector,
            termination_condn=None
            if self.specific_settings["direction"] != "multi-time-reach-back"
            else termination_condn,
        )

        # for open_loop control the times vector must be in absolute times
        times = times + self.current_data_t_0

        if self.specific_settings["direction"] in [
            "backward",
            "multi-time-reach-back",
            "forward-backward",
        ]:
            times = np.flip(times, axis=0)
            x_traj, contr_seq, distr_seq = [
                np.flip(seq, axis=1) for seq in [x_traj, contr_seq, distr_seq]
            ]

        return times, x_traj, contr_seq, distr_seq

    def _log_traj_in_plan_dict(self, times, x_traj, contr_seq):
        """Helper function to log plans throughout the closed-loop control."""
        # Step 1: concatenate to reduce file size
        times_vec = times.reshape(1, -1)
        trajectory = np.concatenate((x_traj, times_vec), axis=0)

        plan_dict = {"traj": trajectory, "ctrl": contr_seq}
        self.planned_trajs.append(plan_dict)

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
            throw_exceptions=True,
        )

        # calculate relative posix_time (we use it in interpolation because jax uses float32 and otherwise cuts off)
        data_xarray = data_xarray.assign(
            relative_time=lambda x: units.get_posix_time_from_np64(x.time)
            - units.get_posix_time_from_np64(data_xarray["time"][0])
        )

        # feed in the current data to the Platform classes
        self.nondim_dynamics.dimensional_dynamics.update_jax_interpolant(data_xarray)

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

    def _get_non_dim_state(self, state: jnp.ndarray):
        """Returns the state transformed from dimensional coordinates to non_dimensional coordinates."""
        return (state.flatten() - self.offset_vec) / self.characteristic_vec

    def _initialize_non_dim_grid(self):
        """Return nondim_grid for the solve."""
        # extract the characteristic scale and offset value for each dimensions
        self.characteristic_vec = self.grid.domain.hi - self.grid.domain.lo
        self.offset_vec = self.grid.domain.lo

        self.nonDimGrid = hj.Grid.nondim_grid_from_dim_grid(
            dim_grid=self.grid,
            characteristic_vec=self.characteristic_vec,
            offset_vec=self.offset_vec,
        )

    def _set_value_func_to_forward_time(self):
        """Arrange class values to forward for easier access: reach_times and all_values."""
        if self.reach_times[0] > self.reach_times[-1]:
            self.reach_times, self.all_values = [
                np.flip(seq, axis=0) for seq in [self.reach_times, self.all_values]
            ]
        else:
            raise ValueError(
                "Reachability Values are already in forward time, this should not happen."
            )

    # PLOTTING FUNCTIONS #
    def plot_reachability_snapshot(
        self,
        rel_time_in_seconds: float = 0,
        ax: plt.Axes = None,
        return_ax: bool = False,
        fig_size_inches: Tuple[int, int] = (12, 12),
        alpha_color: float = 1.0,
        time_to_reach: bool = False,
        granularity_in_h: float = 5,
        plot_in_h: bool = True,
        add_drawing: Callable[[plt.axis, float], None] = None,
        **kwargs,
    ):
        """Plot the reachable set the planner was computing last at  a specific rel_time_in_seconds.
        Args:
            rel_time_in_seconds:    the relative time for which to plot the snapshot since last replan
            ax:                     Optional: axis object to plot on top of
            return_ax:              if true, function returns ax object for more plotting
            fig_size_inches:        Figure size
            ### Rest only relevant for multi-time-reach-back
            alpha_color:            the alpha level of the colors when plotting multi-time-reachability
            time_to_reach:          if True we plot the time-to-reach the target, otherwise the value function
            granularity_in_h:       the granularity of the color-coding
            plot_in_h:              if we want to plot in h (or leave it in seconds)
            ### Other optional arguments
            add_drawing:            A callable to add a drawing to the snapshot, taking in (ax, rel_time_in_seconds)
        """
        if self.grid.ndim != 2:
            raise ValueError("plot_reachability is currently only implemented for 2D sets")

        # create the axis object if not fed in
        if ax is None:
            if self.specific_settings["use_geographic_coordinate_system"]:
                ax = self.last_data_source.set_up_geographic_ax()
            else:
                ax = plt.axes()

        # get_initial_value
        initial_values = self.get_initial_values(direction=self.specific_settings["direction"])

        # interpolate the value function to the specific time
        val_at_t = interp1d(self.reach_times, self.all_values, axis=0, kind="linear")(
            max(
                self.reach_times[0],
                min(self.reach_times[-1], rel_time_in_seconds + self.reach_times[0]),
            )
        ).squeeze()

        # If in normal reachability setting
        is_multi_reach = "multi-time-reach-back" == self.specific_settings["direction"]
        if (
            is_multi_reach and "val_func_levels" not in kwargs
        ):  # value function pre-computations before plotting
            multi_reach_rel_time = (
                rel_time_in_seconds - self.reach_times[-1]
            )  # this is normally negative
            non_dim_val_func_levels, abs_time_y_ticks, y_label = self._get_multi_reach_levels(
                granularity_in_h,
                time_to_reach=time_to_reach,
                vmin=val_at_t.min(),
                abs_time_in_h=multi_reach_rel_time / 3600 if plot_in_h else multi_reach_rel_time,
            )
            # package them in kwargs
            kwargs.update(
                {
                    "val_func_levels": non_dim_val_func_levels,
                    "y_label": y_label,
                    "yticklabels": abs_time_y_ticks,
                }
            )

        # plot the set on top of ax
        ax = hj.viz._visSet2D(
            self.grid,
            val_at_t,
            plot_level=0,
            color_level="black",
            colorbar=is_multi_reach,
            obstacles=None,
            target_set=initial_values,
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
        ax.scatter(
            self.problem.end_region.lon.deg,
            self.problem.end_region.lat.deg,
            color="g",
            marker="x",
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

    def plot_reachability_snapshot_over_currents(
        self, rel_time_in_seconds: float = 0, ax: plt.Axes = None, **kwargs
    ):
        """Plot the reachable set the planner was computing last at  a specific rel_time_in_seconds over the currents.
        Args:
            rel_time_in_seconds:    the relative time for which to plot the snapshot since last replan
            ax:                     Optional: axis object to plot on top of
            kwargs:                 See plot_reachability_snapshot for further arguments
        """
        os.makedirs("generated_media", exist_ok=True)
        # plot currents on ax
        ax = self.last_data_source.plot_data_at_time_over_area(
            time=self.current_data_t_0 + rel_time_in_seconds,
            x_interval=[self.grid.domain.lo[0], self.grid.domain.hi[0]],
            y_interval=[self.grid.domain.lo[1], self.grid.domain.hi[1]],
            return_ax=True,
            colorbar=False,
            ax=ax,
        )
        # add reachability snapshot on top
        return self.plot_reachability_snapshot(
            rel_time_in_seconds=rel_time_in_seconds,
            ax=ax,
            plot_in_h=True,
            display_colorbar=True,
            mask_above_zero=True,
            **kwargs,
        )

    def plot_reachability_animation(
        self,
        time_to_reach: bool = False,
        plot_in_h: bool = True,
        granularity_in_h: int = 1,
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
        **kwargs,
    ):
        """Create an animation of the reachability computation.
        Args:
           time_to_reach:      if True we plot the value function otherwise just the zero level set
           plot_in_h:          if the value function units should be converted to hours
           granularity_in_h:   with which granularity to plot the value function
           filename:           filename under which to save the animation
           temporal_resolution: the temporal resolution in seconds, per default same as data_source
           with_opt_ctrl:      if True the optimal trajectory and control is added as overlay.
           forward_time:       forward_time manually force forward time (otherwise in direction of calculation)
           data_source_for_plt:the data source to plot as background with data_source.animate_data()
           kwargs:             See plot_reachability_snapshot for further arguments (can also add drawings)

        """
        os.makedirs("generated_media", exist_ok=True)
        if "multi-time-reach-back" == self.specific_settings["direction"] and not time_to_reach:
            abs_time_vec = (
                (self.reach_times - self.reach_times[0]) / 3600
                if plot_in_h
                else (self.reach_times - self.reach_times[0])
            )
            non_dim_val_func_levels, abs_time_y_ticks, y_label = self._get_multi_reach_levels(
                granularity_in_h,
                time_to_reach=time_to_reach,
                vmin=self.all_values.min(),
                abs_time_in_h=abs_time_vec[-1],
            )
            # package them in kwargs
            kwargs.update(
                {
                    "val_func_levels": non_dim_val_func_levels,
                    "y_label": y_label,
                    "yticklabels": abs_time_y_ticks,
                }
            )

        def add_reachability_snapshot(ax, time):
            ax = self.plot_reachability_snapshot(
                rel_time_in_seconds=time - self.current_data_t_0,
                granularity_in_h=granularity_in_h,
                alpha_color=1,
                mask_above_zero=True,
                return_ax=True,
                fig_size_inches=(12, 12),
                time_to_reach=time_to_reach,
                ax=ax,
                plot_in_h=plot_in_h,
                display_colorbar=True,
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
            data_source_for_plt = self.last_data_source

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
                add_ax_func=add_reachability_snapshot,
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
                add_reachability_snapshot(ax, temporal_vector[time_idx])

            # set time direction of the animation
            frames_vector = np.where(
                forward_time,
                np.arange(len(temporal_vector)),
                np.flip(np.arange(len(temporal_vector))),
            )
            # create animation function object (it's not yet executed)
            ani = animation.FuncAnimation(fig, func=render_func, frames=frames_vector, repeat=False)

            # render the animation with the keyword arguments
            DataSource.render_animation(animation_object=ani, output=filename, fps=fps)

    def vis_value_func_along_traj(
        self, time_to_reach=False, return_ax=False, plot_in_h=True, figsize=(12, 12)
    ):
        """Plot the Value function along the most recently planned trajectory. Only works for 2D trajectories right now.
        Args:
           time_to_reach:       if True we plot the value function otherwise just the zero level set
           return_ax:           return ax object to plot more on top
           plot_in_h:           if the value function units should be converted to hours
        """
        fig, ax = plt.subplots(figsize=figsize)

        if time_to_reach:
            all_values_dimensional = (
                1 + self.all_values - (self.reach_times / self.reach_times[-1]).reshape(-1, 1, 1)
            )
            all_values = all_values_dimensional * self.specific_settings["T_goal_in_seconds"]
            ylabel = "Earliest-time-to-reach"
        else:
            ylabel = r"$\phi(x_t)$"
            all_values = self.all_values

        reach_times = self.reach_times - self.reach_times[0]

        traj_times = (
            self.planned_trajs[-1]["traj"][2, :] - self.current_data_t_0 - self.reach_times[0]
        )

        if plot_in_h:
            reach_times = reach_times / 3600
            traj_times = traj_times / 3600

        hj.viz.visValFuncTraj(
            ax,
            traj_times=traj_times,
            x_traj=self.planned_trajs[-1]["traj"][:2, :],
            all_times=reach_times,
            all_values=all_values,
            grid=self.grid,
            flip_times=False,
            ylabel=ylabel,
        )

        if return_ax:
            return ax
        else:
            plt.show()

    @staticmethod
    def _get_multi_reach_levels(granularity_in_h, vmin, abs_time_in_h, time_to_reach):
        """Helper function to determine the levels for multi-reachability plotting."""

        n_levels = abs(math.ceil(abs_time_in_h / granularity_in_h)) + 1
        if vmin == 0 or n_levels == 1:
            non_dim_val_func_levels = np.array([0, 1e-10])
            abs_time_y_ticks = np.array([0.0, 0.0])
        else:
            non_dim_val_func_levels = np.linspace(vmin, 0, n_levels)
            abs_time_y_ticks = np.around(np.linspace(abs_time_in_h, 0, n_levels), decimals=0)

        if time_to_reach:
            y_label = "Fastest Time-to-Target in hours"
            abs_time_y_ticks = np.abs(np.flip(abs_time_y_ticks, axis=0))
        else:
            y_label = "HJ Value Function"

        return non_dim_val_func_levels, abs_time_y_ticks, y_label

    ## TTR Value Interpolation (in hours)
    def set_interpolator(self):
        """set interpolator after replaning for quicker interpolation"""
        # Scale TTR Values
        # Step 1: Transform to TTR map in non-dimensional time (T = 1).
        # Formula in paper is: TTR(t) = J + T - t, for us self.all_values.min(axis=(1, 2)) is -(T-t)
        ttr_values = self.all_values - self.all_values.min(axis=(1, 2))[:, None, None]
        # Step 2: After normalization with min, the min should be 0 in all time slices
        # Then multiply with the amount of hours to make it dimensional in time!
        ttr_values = (
            ttr_values
            / np.abs(self.all_values.min())
            * np.abs(self.reach_times[-1] - self.reach_times[0])
            / 3600
        )

        # Set Interpolator for quicker interpolation
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            points=(
                self.current_data_t_0 + self.reach_times,
                self.grid.coordinate_vectors[0],
                self.grid.coordinate_vectors[1],
            ),
            values=ttr_values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def interpolate_value_function_in_hours(
        self,
        observation: ArenaObservation = None,
        point: SpatioTemporalPoint = None,
        width_deg: Optional[float] = 0,
        width: Optional[int] = 1,
        allow_spacial_extrapolation: Optional[bool] = False,
        allow_temporal_extrapolation: Optional[bool] = False,
    ) -> np.ndarray:
        """
        Get interpolated TTR value in hours either for a single point or on a grid (if width's passed)
        Arguments:
            observation: observation to get value at/around,checks for replaning
            point: pointt to get value at/around
            width_deg: width in degrees for the grid
            width: width in points for the grid
        """
        if observation is not None and isinstance(observation, ArenaObservation):
            self.replan_if_necessary(observation)
            point = observation.platform_state.to_spatio_temporal_point()

        if not type(point) in [SpatioTemporalPoint, PlatformState]:
            raise Exception(
                "Either ArenaObservation or SpatioTemporalPoint has to be given for interpolation."
            )

        out_x = np.linspace(point.lon.deg - width_deg / 2, point.lon.deg + width_deg / 2, width)
        out_y = np.linspace(point.lat.deg - width_deg / 2, point.lat.deg + width_deg / 2, width)
        out_t = point.date_time.timestamp()

        # Sanitize Inputs:
        extrapolate_x = (
            out_x[0] <= self.grid.coordinate_vectors[0][0]
            or self.grid.coordinate_vectors[0][-1] <= out_x[-1]
        )
        extrapolate_y = (
            out_y[0] <= self.grid.coordinate_vectors[1][0]
            or self.grid.coordinate_vectors[1][-1] <= out_y[-1]
        )
        extrapolate_t = not (
            self.current_data_t_0 + self.reach_times[0]
            <= out_t
            <= self.current_data_t_0 + self.reach_times[-1]
        )
        if extrapolate_x or extrapolate_y or extrapolate_t:
            message = (
                f"Extrapolating in {'x' if extrapolate_x else 'y' if extrapolate_y else 't'}. "
                + f"Requested: out_t: {out_t:.0f} out_x: [{out_x[0]:.2f}, {out_x[-1]:.2f}] out_y: [{out_y[0]:.2f}, {out_y[-1]:.2f}]. "
                + "Available:"
                + f" in_t: [{self.current_data_t_0+self.reach_times[0]:.0f}, {self.current_data_t_0+self.reach_times[-1]:.0f}]"
                + f" in_x: [{self.grid.coordinate_vectors[0][0]:.2f}, {self.grid.coordinate_vectors[0][-1]:.2f}]"
                + f" in_y: [{self.grid.coordinate_vectors[1][0]:.2f}, {self.grid.coordinate_vectors[1][-1]:.2f}]"
            )
            if (allow_spacial_extrapolation and not extrapolate_t) or (
                allow_temporal_extrapolation and not (extrapolate_x or extrapolate_y)
            ):
                self.logger.warning(message)
            else:
                raise ValueError(message)

        mx, my = np.meshgrid(out_x, out_y)

        return (
            self.interpolator((np.repeat(out_t, my.size), mx.ravel(), my.ravel()))
            .reshape((width, width))
            .squeeze()
        )

    ## Saving & Loading Planner State
    def save_planner_state(self, folder):
        os.makedirs(folder, exist_ok=True)

        # Settings
        with open(folder + "specific_settings.pickle", "wb") as file:
            pickle.dump(self.specific_settings, file)

        # Used in Replanning
        with open(folder + "last_fmrc_time_planned_with.pickle", "wb") as file:
            pickle.dump(self.last_fmrc_time_planned_with, file)
        with open(folder + "planner_cache_index.pickle", "wb") as file:
            pickle.dump(self.planner_cache_index, file)
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
        with open(folder + "initial_values.pickle", "wb") as file:
            pickle.dump(self.initial_values, file)

        self.logger.info(f"HJPlannerBase: Saving plan to {folder}")

    def pickle(self, dir):
        with open(dir, "wb") as f:
            pickle.dump(self, f)

    def restore_state(self, folder):
        # Used in Replanning
        with open(folder + "last_fmrc_time_planned_with.pickle", "rb") as file:
            self.last_fmrc_time_planned_with = pickle.load(file)
        with open(folder + "planner_cache_index.pickle", "rb") as file:
            self.planner_cache_index = pickle.load(file)
        # Used in Interpolation
        with open(folder + "all_values.pickle", "rb") as file:
            self.all_values = pickle.load(file)
        with open(folder + "reach_times.pickle", "rb") as file:
            self.reach_times = pickle.load(file)
        with open(folder + "grid.pickle", "rb") as file:
            self.grid = pickle.load(file)
        with open(folder + "current_data_t_0.pickle", "rb") as file:
            self.current_data_t_0 = pickle.load(file)
        with open(folder + "current_data_t_T.pickle", "rb") as file:
            self.current_data_t_T = pickle.load(file)
        # Used in Start Sampling
        with open(folder + "characteristic_vec.pickle", "rb") as file:
            self.characteristic_vec = pickle.load(file)
        with open(folder + "initial_values.pickle", "rb") as file:
            self.initial_values = pickle.load(file)

        self.set_interpolator()

        self.logger.info(
            f"HJPlannerBase: Plan loaded from {folder} with fmrc_time={self.last_fmrc_time_planned_with}"
        )
