import math
import numpy as np
import abc
import xarray as xr
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformAction
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatioTemporalPoint, SpatialPoint
from ocean_navigation_simulator.env.controllers.Controller import Controller
from ocean_navigation_simulator.env.utils import units
from typing import Tuple, Optional, Dict, List, AnyStr, Union, Callable
# from ocean_navigation_simulator.utils import simulation_utils
from jax.interpreters import xla
import jax.numpy as jnp
from functools import partial
from scipy.interpolate import interp1d
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import warnings
import math
import sys
# Note: if you develop on hj_reachability and this library simultaneously uncomment this line
# sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) + 'hj_reachability_c3'])
import hj_reachability as hj


# TODO: handle where it is using hours_to_hj_solve_timescale to make sure the plot is in hours
# TODO: This is very much work in progress, does not work yet!


class HJPlannerBase(Controller):
    """
    Baseclass for all HJ reachability-based Planners using backwards/forwards/multi-time reachability.
        For details see: "A future for intelligent autonomous ocean observing systems" P.F.J. Lermusiaux

        Note: The Baseclass is general and works for 2D, 3D, 4D System.
        In the Baseclass, the PDE is solved in non_dimensional dynamics in space and time to reduce numerical errors.
        To use this class, only the 'abstractmethod' functions need to be implemented.

        See Planner class for the rest of the attributes.
    """

    def __init__(self, problem: Problem, specific_settings: Dict):
        """
        Constructor for the HJ Planner Baseclass.
        Args:
            problem: the Problem the controller will run on
            specific_settings: Attributes required in the specific_settings dict
                direction: string of {'forward', 'backward', 'forward-backward', 'multi-reach-back}
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
        super().__init__(problem, specific_settings)

        # create a variable that persists across runs of self.plan() to reference the currently reload data
        self.current_data_t_0, self.current_data_t_T = [None] * 2
        # Two variables to enable both re-planning with fixed frequency and only when new forecast available
        self.last_data_source, self.last_fmrc_idx_planned_with, self.last_planning_posix = [None] * 3

        # this is just a variable that persists after planning for plotting/debugging
        self.x_t = None

        # Initializes variables needed when solving the HJ PDE, they will be filled in the plan method
        self.reach_times, self.all_values, self.grid = [None] * 3
        self.diss_scheme = self._get_dissipation_schema()
        self.x_traj, self.contr_seq, self.distr_seq = [None] * 3
        # Initialize variables needed for solving the PDE in non_dimensional terms
        self.characteristic_vec, self.offset_vec, self.nonDimGrid, self.nondim_dynamics = [None] * 4

        # Initialize the non_dimensional_dynamics and within it the dimensional_dynamics
        # Note: as initialized here, it's not usable, only after 'update_current_data' is called for the first time.
        self.nondim_dynamics = hj.dynamics.NonDimDynamics(dimensional_dynamics=self.get_dim_dynamical_system())

        if self.specific_settings['d_max'] > 0 and self.specific_settings['direction'] == "multi-time-reach-back":
            print("No disturbance implemented for multi-time reachability, only runs with d_max=0.")

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """ Main interface function for the simulation, so all the logic to trigger re-planning is inside here.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        Returns:
            PlatformAction dataclass
        """
        # Step 1: Check if we should re-plan based on specified criteria
        if self._check_for_replanning(observation):
            print("Reachability Planner: Planning")
            # log x_t and data_source for plotting and easier access later
            self.x_t = observation.platform_state
            self.last_data_source = observation.forecast_data_source
            # Update the data
            self._update_current_data(observation=observation)
            self._plan(observation.platform_state)

        # Step 2: return the action from the plan
        return self._get_action_from_plan(state=observation.platform_state)

    def _check_for_replanning(self, observation: ArenaObservation) -> bool:
        """Helper Function to check if we want to replan with HJ Reachability.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        """
        # For the first round for sure
        if self.last_fmrc_idx_planned_with is None:
            old = self.last_fmrc_idx_planned_with
            # data and logging variables need to be initialized at first round
            self.last_fmrc_idx_planned_with = observation.forecast_data_source.check_for_most_recent_fmrc_dataframe(
                time=observation.platform_state.date_time)
            print(f'No Forecast Index (Old: {old}, New: {self.last_fmrc_idx_planned_with}).')
            return True
        # Check for re-planning with new forecast
        elif self.specific_settings['replan_on_new_fmrc']:
            if self._new_forecast_data_available(observation):
                print('New Forecast available.')
                return True
        # Check for re-planning after fixed time intervals
        elif self.specific_settings['replan_every_X_seconds'] is not None:
            if self.last_planning_posix + self.specific_settings['replan_every_X_seconds'] \
                    >= observation.platform_state.date_time.timestamp():
                print('Periodic Replanning.')
                return True

        return False

    def _new_forecast_data_available(self, observation: ArenaObservation) -> bool:
        """Helper function to check if new forecast data is available in the forecast_data_source.
        If yes, the data in the interpolation is updated and True is returned, otherwise False.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        """
        # Get the idx for the most recent file
        most_current_fmrc_idx_at_time = observation.forecast_data_source.check_for_most_recent_fmrc_dataframe(
            time=observation.platform_state.date_time)
        # Check if this is after our last planned one
        if most_current_fmrc_idx_at_time != self.last_fmrc_idx_planned_with:
            # update the current data in the jax interpolatn
            self.last_fmrc_idx_planned_with = most_current_fmrc_idx_at_time
            return True
        else:
            return False

    def _get_action_from_plan(self, state: PlatformState) -> PlatformAction:
        """ Extracts the next action from the most recent plan (saved value function).
            if specific_settings['direction'] is forward:     applying the actions open-loop from the contr_seq
            if specific_settings['direction'] is backward:    computing the gradient/action directly from the value function

            Args:
                state: PlatformState containing the location & time
            Returns:
                PlatformAction to send to the simulation
            """
        if self.specific_settings['direction'] == 'forward':
            u_out = super().get_open_loop_control_from_plan(state=state)
        else:
            # check if time is outside times and throw warning if yes but continue.
            rel_time = state.date_time.timestamp() - self.current_data_t_0
            if rel_time > self.reach_times[-1]:
                warnings.warn("HJPlanner Warning: Extrapolating time beyond the reach_times, should replan.",
                              RuntimeWarning)
                rel_time = self.reach_times[-1]

            # Extract the optimal control from the calculated value function at the specific platform state.
            u_out, _ = self.nondim_dynamics.dimensional_dynamics.get_opt_ctrl_from_values(
                grid=self.grid, x=self.get_x_from_full_state(state),
                time=rel_time,
                times=self.reach_times, all_values=self.all_values)

        return PlatformAction(magnitude=u_out[0], direction=u_out[1])

    def get_waypoints(self) -> List[SpatioTemporalPoint]:
        """Returns: a list of waypoints each containing [lon, lat, time]"""
        return np.vstack((self.x_traj, self.times)).T.tolist()

    def _get_dissipation_schema(self):
        """Helper function to directly set the artificial dissipation schema used in solving the PDE."""
        # Note: can be done more elegantly by list_indexing =)
        if self.specific_settings['artificial_dissipation_scheme'] == 'local_local':
            return hj.artificial_dissipation.local_local_lax_friedrichs
        elif self.specific_settings['artificial_dissipation_scheme'] == 'local':
            return hj.artificial_dissipation.local_lax_friedrichs
        elif self.specific_settings['artificial_dissipation_scheme'] == 'global':
            return hj.artificial_dissipation.global_lax_friedrichs
        else:
            raise ValueError("artificial_dissipation_scheme is not one of {global, local, local_local}")

    @abc.abstractmethod
    def initialize_hj_grid(self, xarray: xr):
        """ Initialize grid to solve PDE on."""
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
            raise ValueError(
                "Current time {} is before the start of the forecast data. This should not happen. current_data_t_0 is {}".format(
                    x_t.date_time), datetime.fromtimestamp(self.current_data_t_0, tz=timezone.utc))
        # Check if the current_data is sufficient for planning over the specified time horizon.
        # if not enough we throw a warning
        if x_t.date_time.timestamp() + self.specific_settings['T_goal_in_seconds'] > self.current_data_t_T:
            warnings.warn(
                "Loaded forecast data does not contain the full time-horizon from x_t {} to T_goal_in_seconds {}. Automatically adjusting.".format(
                    x_t.date_time, self.specific_settings['T_goal_in_seconds']))

    def _plan(self, x_t: PlatformState):
        """Main function where the reachable front is computed.
        Args:
            x_t: Platform state used as start/target of HJ Reachability computation, depending on 'direction'.
        """

        # run data checks if the right current data is loaded in the interpolation function
        self._check_data_settings_compatibility(x_t=x_t)

        # Step 2: depending on the reachability direction run the respective algorithm
        if self.specific_settings['direction'] == 'forward':
            self._run_hj_reachability(initial_values=self.get_initial_values(direction="forward"),
                                      t_start=x_t.date_time,
                                      T_max_in_seconds=self.specific_settings['T_goal_in_seconds'],
                                      dir='forward',
                                      x_reach_end=self.get_x_from_full_state(self.problem.end_region))
            self._extract_trajectory(x_start=self.get_x_from_full_state(self.problem.end_region))

        elif self.specific_settings['direction'] == 'backward':
            # Note: no trajectory is extracted as the value function is used for closed-loop control
            self._run_hj_reachability(initial_values=self.get_initial_values(direction="backward"),
                                      t_start=x_t.date_time,
                                      T_max_in_seconds=self.specific_settings['T_goal_in_seconds'],
                                      dir='backward')
            self._extract_trajectory(x_start=self.get_x_from_full_state(x_t))
            # arrange to forward times by convention for plotting and open-loop control
            self._flip_value_func_to_forward_times()

        elif self.specific_settings['direction'] == 'forward-backward':
            # Step 1: run the set forward to get the earliest possible arrival time
            self._run_hj_reachability(initial_values=self.get_initial_values(direction="forward"),
                                      t_start=x_t.date_time,
                                      T_max_in_seconds=self.specific_settings['T_goal_in_seconds'],
                                      dir='forward',
                                      x_reach_end=self.get_x_from_full_state(self.problem.end_region))
            # Step 2: run the set backwards from the earliest arrival time backwards
            _, t_earliest_in_seconds = self._get_t_earliest_for_target_region()
            print("earliest for target region is ", t_earliest_in_seconds)
            self._run_hj_reachability(initial_values=self.get_initial_values(direction="backward"),
                                      t_start=x_t.date_time,
                                      T_max_in_seconds=t_earliest_in_seconds + self.specific_settings[
                                          'fwd_back_buffer_in_seconds'],
                                      dir='backward')
            self._extract_trajectory(x_start=self.get_x_from_full_state(x_t))
            # arrange to forward times by convention for plotting and open-loop control
            self._flip_value_func_to_forward_times()
        elif self.specific_settings['direction'] == 'multi-time-reach-back':
            # Step 1: run multi-reachability backwards in time
            self._run_hj_reachability(initial_values=self.get_initial_values(direction="multi-time-reach-back"),
                                      t_start=x_t.date_time,
                                      T_max_in_seconds=self.specific_settings['T_goal_in_seconds'],
                                      dir='multi-time-reach-back')

            # Now just extract it forwards releasing the vehicle at t=0
            def termination_condn(x_target, r, x, t):
                return np.linalg.norm(x_target - x) <= r

            termination_condn = partial(termination_condn, jnp.array(self.problem.end_region),
                                        self.problem.target_radius)
            self._extract_trajectory(self.get_x_from_full_state(x_t), termination_condn=termination_condn)
            # arrange to forward times by convention for plotting and open-loop control (aka closed-loop with this)
            self._flip_value_func_to_forward_times()
            if self.all_values.min() < -2:
                raise ValueError(
                    "HJPlanner: Some issue with the value function, min goes below -2, should maximally be -1.")
        else:
            raise ValueError(
                "Direction in controller YAML needs to be one of {backward, forward, forward-backward, "
                "multi-time-reach-back}")

        # check if all_values contains any Nans
        if jnp.isnan(self.all_values).sum() > 0:
            raise ValueError("HJ Planner has NaNs in all values. Something went wrong in solving the PDE.")

        self.last_planning_posix = x_t.date_time.timestamp()

    def _run_hj_reachability(self, initial_values: jnp.ndarray, t_start: datetime, T_max_in_seconds: float,
                             dir: AnyStr, x_reach_end: jnp.ndarray = None):
        """ Run hj reachability starting with initial_values at t_start for maximum of T_max_in_seconds
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
        self.nondim_dynamics.tau_c = min(T_max_in_seconds, int(self.current_data_t_T - self.current_data_t_0))
        # This is in relative seconds since current data t_0
        self.nondim_dynamics.t_0 = t_start.timestamp() - self.current_data_t_0

        # set up the non_dimensional time-vector for which to save the value function
        solve_times = np.linspace(0, 1, self.specific_settings['n_time_vector'] + 1)

        if dir == 'backward' or dir == 'multi-time-reach-back':
            solve_times = np.flip(solve_times, axis=0)
            self.nondim_dynamics.dimensional_dynamics.control_mode = 'min'
            self.nondim_dynamics.dimensional_dynamics.disturbance_mode = 'max'
        elif dir == 'forward':
            self.nondim_dynamics.dimensional_dynamics.control_mode = 'max'
            self.nondim_dynamics.dimensional_dynamics.disturbance_mode = 'min'

        # specific settings for multi-time-reach-back
        if dir == 'multi-time-reach-back':
            # write multi_reach hamiltonian postprocessor
            def multi_reach_step(mask, val):
                val = jnp.where(mask <= 0, -1, val)
                return val

            # combine it with partial sp the mask input gets fixed and only val is open
            p_multi_reach_step = partial(multi_reach_step, initial_values)
            # set the postprocessor to be fed into solver_settings
            hamiltonian_postprocessor = p_multi_reach_step
            print("running multi-reach")
        else:  # make the postprocessor the identity
            hamiltonian_postprocessor = lambda *x: x[-1]

        # create solver settings object
        solver_settings = hj.SolverSettings.with_accuracy(accuracy=self.specific_settings['accuracy'],
                                                          x_init=self._get_non_dim_state(
                                                              x_reach_end) if x_reach_end is not None else None,
                                                          artificial_dissipation_scheme=self.diss_scheme,
                                                          hamiltonian_postprocessor=hamiltonian_postprocessor)

        # solve the PDE in non_dimensional to get the value function V(s,t)
        non_dim_reach_times, self.all_values = hj.solve(
            solver_settings=solver_settings,
            dynamics=self.nondim_dynamics,
            grid=self.nonDimGrid,
            times=solve_times,
            initial_values=initial_values,
            progress_bar=self.specific_settings['progress_bar']
        )

        # scale up the reach_times to be dimensional_times in seconds again
        self.reach_times = non_dim_reach_times * self.nondim_dynamics.tau_c + self.nondim_dynamics.t_0

    def _get_t_earliest_for_target_region(self) -> Tuple:
        """Helper Function to get the earliest time the forward reachable set overlaps with the target region."""
        # get target_region_mask
        target_region_mask = self.get_initial_values(direction="backward") <= 0

        # iterate forward to get earliest time it's inside
        for idx in range(self.all_values.shape[0]):
            reached = np.logical_and(target_region_mask, self.all_values[idx, ...] <= 0).any()
            if reached:
                break
        # extract earliest relative time of idx
        T_earliest_in_seconds = self.reach_times[idx] - self.reach_times[0]
        if not reached:
            print("Not reached, returning maximum time for the backwards reachability.")
        return reached, T_earliest_in_seconds

    def _extract_trajectory(self, x_start: jnp.ndarray, traj_rel_times_vector: jnp.ndarray = None,
                            termination_condn: Callable = None):
        """Backtrack the reachable front to extract a trajectory etc.

        Args:
        x_start:                   start_point for backtracking must be same dim as grid.ndim
        traj_rel_times_vector:     the times vector for which to extract trajectory points for
                                    in seconds from the start of the reachability computation t=0.
                                    Defaults to self.reach_times.
        termination_condn:          function to evaluate if the calculation should be terminated (e.g. because reached)
        """
        # setting default times vector for the trajectory
        if traj_rel_times_vector is None:
            traj_rel_times_vector = self.reach_times
        else:
            traj_rel_times_vector = traj_rel_times_vector + self.reach_times[0]

        self.times, self.x_traj, self.contr_seq, self.distr_seq = \
            self.nondim_dynamics.dimensional_dynamics.backtrack_trajectory(
                grid=self.grid, x_init=x_start, times=self.reach_times, all_values=self.all_values,
                traj_times=traj_rel_times_vector, termination_condn=termination_condn)

        # for open_loop control the times vector must be in absolute times
        self.times = self.times + self.current_data_t_0

        if self.specific_settings['direction'] in ['backward', 'multi-time-reach-back', 'forward-backward']:
            self._flip_traj_to_forward_times()

        # log the planned trajectory for later inspection purpose
        # Step 1: concatenate to reduce file size
        times_vec = self.times.reshape(1, -1)
        trajectory = np.concatenate((self.x_traj, np.ones(times_vec.shape), times_vec), axis=0)

        plan_dict = {'traj': trajectory, 'ctrl': self.contr_seq}
        self.planned_trajs.append(plan_dict)

    def _update_current_data(self, observation: ArenaObservation):
        """Helper function to load new current data into the interpolation.
        Args:
            observation: observation returned by the simulator (containing the forecast_data_source)
        """
        print("Reachability Planner: Loading new current data.")

        # Step 1: get the x,y,t bounds for current position, goal position and settings.
        t_interval, y_interval, x_interval = observation.forecast_data_source.convert_to_x_y_time_bounds(
            x_0=observation.platform_state.to_spatio_temporal_point(), x_T=self.problem.end_region,
            deg_around_x0_xT_box=self.specific_settings['deg_around_xt_xT_box'],
            temp_horizon_in_s=self.specific_settings['T_goal_in_seconds'])

        # get the data subset from the file
        data_xarray = observation.forecast_data_source.get_data_over_area(
            x_interval=x_interval, y_interval=y_interval, t_interval=t_interval,
            spatial_resolution=self.specific_settings['grid_res'])

        # calculate relative posix_time (we use it in interpolation because jax uses float32 and otherwise cuts off)
        data_xarray = data_xarray.assign(
            relative_time=lambda x: units.get_posix_time_from_np64(x.time) - units.get_posix_time_from_np64(
                data_xarray['time'][0]))

        # feed in the current data to the Platform classes
        self.nondim_dynamics.dimensional_dynamics.update_jax_interpolant(data_xarray)

        # set absolute time in UTC Posix time
        self.current_data_t_0 = units.get_posix_time_from_np64(data_xarray['time'][0]).data
        # set absolute final time in UTC Posix time
        self.current_data_t_T = units.get_posix_time_from_np64(data_xarray['time'][-1]).data

        # initialize the grids and dynamics to solve the PDE with
        self.initialize_hj_grid(data_xarray)
        self._initialize_non_dim_grid()
        # update non_dimensional_dynamics with the new non_dim scaling and offset
        self.nondim_dynamics.characteristic_vec = self.characteristic_vec
        self.nondim_dynamics.offset_vec = self.offset_vec

        # Delete the old caches (might not be necessary for analytical fields -> investigate)
        # print("Cache Size: ", hj.solver._solve._cache_size())
        hj.solver._solve._clear_cache()
        xla._xla_callable.cache_clear()

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

    def _get_non_dim_state(self, state: jnp.ndarray):
        """Returns the state transformed from dimensional coordinates to non_dimensional coordinates."""
        return (state.flatten() - self.offset_vec) / self.characteristic_vec

    def _initialize_non_dim_grid(self):
        """ Return nondim_grid for the solve."""
        # extract the characteristic scale and offset value for each dimensions
        self.characteristic_vec = self.grid.domain.hi - self.grid.domain.lo
        self.offset_vec = self.grid.domain.lo

        self.nonDimGrid = hj.Grid.nondim_grid_from_dim_grid(
            dim_grid=self.grid, characteristic_vec=self.characteristic_vec, offset_vec=self.offset_vec)

    def _flip_traj_to_forward_times(self):
        """ Arrange traj class values to forward for easier access: traj_times, x_traj, contr_seq, distr_seq"""
        # arrange everything forward in time for easier access if we ran it backwards
        if self.times[0] > self.times[-1]:
            self.times = np.flip(self.times, axis=0)
            self.x_traj, self.contr_seq, self.distr_seq = \
                [np.flip(seq, axis=1) for seq in [self.x_traj, self.contr_seq, self.distr_seq]]
        else:
            raise ValueError("Trajectory is already in forward time.")

    def _flip_value_func_to_forward_times(self):
        """ Arrange class values to forward for easier access: reach_times and all_values."""
        if self.reach_times[0] > self.reach_times[-1]:
            self.reach_times, self.all_values = [np.flip(seq, axis=0) for seq in
                                                 [self.reach_times, self.all_values]]
        else:
            raise ValueError("Reachability Values are already in forward time.")

    def plot_reachability_snapshot(self, rel_time_in_seconds: float, input_ax: plt.Axes = None,
                                   return_ax: bool = False, fig_size_inches: Tuple[int, int] = (12, 12),
                                   alpha_color: float = 1., time_to_reach: bool = False,
                                   granularity_in_h: float = 5, plot_in_h: bool = True, **kwargs):
        """ Plot the reachable set the planner was computing last at  a specific rel_time_in_seconds.
        Args:
            rel_time_in_seconds:    the relative time for which to plot the snapshot since last replan
            input_ax:               axis object to plot on top of
            return_ax:              if true, function returns ax object for more plotting
            fig_size_inches:        Figure size
            ### Rest only relevant for multi-time-reach-back
            alpha_color:            the alpha level of the colors when plotting multi-time-reachability
            time_to_reach:          if True we plot the time-to-reach the target, otherwise the value function
            granularity_in_h:       the granularity of the color-coding
            plot_in_h:              if we want to plot in h (or leave it in seconds)
        """
        if self.grid.ndim != 2:
            raise ValueError("plot_reachability is currently only implemented for 2D sets")

        # get_initial_value
        initial_values = self.get_initial_values(direction=self.specific_settings['direction'])

        # interpolate
        val_at_t = interp1d(self.reach_times - self.reach_times[0], self.all_values, axis=0, kind='linear')(
            rel_time_in_seconds).squeeze()

        # If in normal reachability setting
        is_multi_reach = 'multi-time-reach-back' == self.specific_settings['direction']
        if is_multi_reach:  # some pre-computations before plotting
            multi_reach_rel_time = rel_time_in_seconds - self.reach_times[-1]
            non_dim_val_func_levels, abs_time_y_ticks, y_label = self._get_multi_reach_levels(
                granularity_in_h, time_to_reach=time_to_reach, vmin=val_at_t.min(),
                abs_time_in_h=multi_reach_rel_time / 3600 if plot_in_h else multi_reach_rel_time)
            # package them in kwargs
            kwargs.update(
                {'val_func_levels': non_dim_val_func_levels, 'y_label': y_label, 'yticklabels': abs_time_y_ticks})

        if self.last_data_source.source_config_dict['use_geographic_coordinate_system'] and input_ax is None:
            input_ax = self.last_data_source.set_up_geographic_ax()

        # plot the set
        ax = hj.viz._visSet2D(self.grid, val_at_t, plot_level=0, color_level='black',
                              colorbar=is_multi_reach, obstacles=None, target_set=initial_values, return_ax=True,
                              input_ax=input_ax, alpha_colorbar=alpha_color, **kwargs)

        ax.scatter(self.problem.start_state.lon.deg, self.problem.start_state.lat.deg, color='r', marker='o')
        ax.scatter(self.problem.end_region.lon.deg, self.problem.end_region.lat.deg, color='g', marker='x')

        if self.specific_settings['use_geographic_coordinate_system']:
            ax.set_title("Multi-Reach at time {}".format(datetime.fromtimestamp(
                self.reach_times[0] + rel_time_in_seconds + self.current_data_t_0,
                tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')))
        else:
            ax.set_title("Multi-Reach at time {} hours".format(
                self.reach_times[0] + rel_time_in_seconds + self.current_data_t_0))

        # adjust the fig_size
        fig = plt.gcf()
        fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
        if return_ax:
            return ax
        else:
            plt.show()

    def plot_reachability_animation(self, type='mp4', granularity_in_h=5, time_to_reach=False, plot_in_h: bool = True):
        """Create an animation of the reachability computation."""
        # Some pre-computation to adapt the plots accordingly
        is_multi_reach = 'multi-time-reach-back' == self.specific_settings['direction']
        abs_time_vec = (self.reach_times - self.reach_times[0]) / 3600 if plot_in_h else (
                    self.reach_times - self.reach_times[0])
        kwargs = {}
        if is_multi_reach:
            non_dim_val_func_levels, abs_time_y_ticks, y_label = self._get_multi_reach_levels(
                granularity_in_h, time_to_reach=time_to_reach, vmin=self.all_values.min(),
                abs_time_in_h=abs_time_vec[-1])
            # package them in kwargs
            kwargs = {'val_func_levels': non_dim_val_func_levels, 'y_label': y_label,
                      'color_yticklabels': abs_time_y_ticks}

        # Determine x_init and direction
        if self.specific_settings['direction'] == 'forward':
            x_init = np.array(self.problem.end_region)[:2]
            values_to_plot = self.all_values
        else:  # backwards
            x_init = np.array(self.x_t)[:2]
            abs_time_vec = jnp.flip(abs_time_vec)
            values_to_plot = jnp.flip(self.all_values, axis=0)

        # create the animation
        hj.viz.visSet2DAnimation(
            self.grid, values_to_plot, times=abs_time_vec, filename="2D_reachability_animation",
            type=type, x_init=x_init, colorbar=is_multi_reach, **kwargs)

    def vis_value_func_along_traj(self, figsize=(12, 12), return_ax=False, extra_traj=None, time_to_reach=False):
        """Plot the Value function along the most recently planned trajectory."""
        fig, ax = plt.subplots(figsize=figsize)

        if time_to_reach:
            all_values_dimensional = 1 + self.all_values - (self.reach_times / self.reach_times[-1]).reshape(-1, 1,
                                                                                                             1)
            all_values = all_values_dimensional * self.specific_settings['T_goal_in_seconds']
            ylabel = "Earliest-time-to-reach"
        else:
            ylabel = r"$\phi(x_t)$"
            all_values = self.all_values

        reach_times = (self.reach_times - self.reach_times[0]) / self.specific_settings[
            'hours_to_hj_solve_timescale']
        traj_times = (self.planned_trajs[-1]['traj'][3, :] - self.current_data_t_0 - self.reach_times[0]) / \
                     self.specific_settings['hours_to_hj_solve_timescale']

        hj.viz.visValFuncTraj(ax,
                              traj_times=traj_times,
                              x_traj=self.planned_trajs[-1]['traj'][:2, :],
                              all_times=reach_times,
                              all_values=all_values, grid=self.grid,
                              flip_times=False,
                              ylabel=ylabel)
        if extra_traj is not None:
            extra_traj_times = (extra_traj[3, :] - self.current_data_t_0 - self.reach_times[0]) / \
                               self.specific_settings['hours_to_hj_solve_timescale']
            hj.viz.visValFuncTraj(ax,
                                  traj_times=extra_traj_times,
                                  x_traj=extra_traj[:2, :],
                                  all_times=self.reach_times / self.specific_settings[
                                      'hours_to_hj_solve_timescale'],
                                  all_values=all_values, grid=self.grid,
                                  flip_times=False,
                                  ylabel=ylabel)
        if return_ax:
            return ax
        else:
            plt.show()

    @staticmethod
    def _get_multi_reach_levels(granularity_in_h, vmin, abs_time_in_h, time_to_reach):
        """Helper function to determine the levels for multi-reachability plotting."""

        n_levels = abs(math.ceil(abs_time_in_h / granularity_in_h)) + 1
        non_dim_val_func_levels = np.linspace(vmin, 0, n_levels)
        abs_time_y_ticks = np.around(np.linspace(abs_time_in_h, 0, n_levels), decimals=0)

        if time_to_reach:
            y_label = 'Fastest Time-to-Target in hours'
            abs_time_y_ticks = np.abs(np.flip(abs_time_y_ticks, axis=0))
        else:
            y_label = 'HJ Value Function'

        return non_dim_val_func_levels, abs_time_y_ticks, y_label
