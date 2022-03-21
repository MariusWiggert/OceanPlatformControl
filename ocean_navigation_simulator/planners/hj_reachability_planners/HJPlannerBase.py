from ocean_navigation_simulator.planners.planner import Planner
import numpy as np
from ocean_navigation_simulator.utils import simulation_utils
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


class HJPlannerBase(Planner):
    """Baseclass for all HJ reachability-based Planners using backwards/forwards reachability.
        For details see: "A future for intelligent autonomous ocean observing systems" P.F.J. Lermusiaux

        Note: The Baseclass is general and works for 2D, 3D, 4D System.
        In the Baseclass, the PDE is solved in non_dimensional dynamics in spate and time to reduce numerical errors.
        To use this class, only the 'abstractmethod' functions need to be implemented.

        Attributes required in the specific_settings dict
                direction: string of {'forward', 'backward', 'forward-backward'}
                    If forward or backwards reachability should be run.
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

        See Planner class for the rest of the attributes.
    """
    def __init__(self, problem, specific_settings, conv_m_to_deg):
        # initialize Planner superclass
        super().__init__(problem, specific_settings, conv_m_to_deg)

        # create a variable that persists across runs of self.plan() to reference the currently reload data
        self.current_data_t_0, self.current_data_t_T = [None] * 2
        # this is just a variable that persists after planning for plotting/debugging
        self.x_t = None
        # initializes variables needed for planning, they will be filled in the plan method
        self.reach_times, self.all_values, self.grid, self.diss_scheme = [None] * 4
        self.x_traj, self.contr_seq, self.distr_seq = [None] * 3
        # initialize variables needed for solving the PDE in non_dimensional terms
        self.characteristic_vec, self.offset_vec, self.nonDimGrid, self.nondim_dynamics = [None] * 4
        self.set_diss_schema()

        # Initialize the non_dimensional_dynamics and within it the dimensional_dynamics
        # Note: as initialized here, it's not usable, only after 'update_current_data' is called for the first time.
        self.nondim_dynamics = hj.dynamics.NonDimDynamics(dimensional_dynamics=self.get_dim_dynamical_system())

    # abstractmethod: needs to be implemented for each planner
    def initialize_hj_grid(self, grids_dict):
        """ Initialize grid to solve PDE on. """
        raise ValueError("initialize_hj_grid needs to be implemented in child class")

    # abstractmethod: needs to be implemented for each planner
    def get_dim_dynamical_system(self):
        """Creates the dimensional dynamics object and returns it."""
        raise ValueError("get_dim_dynamical_system must be implemented in the child class")

    # abstractmethod: needs to be implemented for each planner
    def get_initial_values(self, direction):
        """Create the initial value function over the grid must be implemented by specific planner."""
        raise ValueError("get_initial_values must be implemented in the child class")

    # abstractmethod: needs to be implemented for each planner
    def get_x_from_full_state(self, x):
        """Return the x state appropriate for the specific reachability planner."""
        raise ValueError("get_x_start must be implemented in the child class")

    def plan(self, x_t, trajectory=None):
        """Main function where the reachable front is computed."""
        # Step 1: read the relevant subset of data (if it changed)
        if self.updated_forecast_source:
            self.update_current_data(x_t=x_t)

        # Check if x_t is in the forecast times and transform to rel_time in seconds
        if x_t[3] < self.current_data_t_0:
            raise ValueError("Current time {} is before the start of the forecast file. This should not happen. t_range {}".format(
                datetime.utcfromtimestamp(x_t[3][0])), self.forecast_data_source['grid_dict']['t_range'])
        # Check if the current_data is sufficient for planning over the specified time horizon, if not give warning.
        if x_t[3] + self.specific_settings['hours_to_hj_solve_timescale'] * self.specific_settings['T_goal_in_h'] > self.current_data_t_T:
            warnings.warn("Forecast file {} with range {} does not contain the full time-horizon from x_t {} to T_goal_in_h {}. Automatically adjusting.".format(
                self.forecast_data_source['content'][self.forecast_data_source['current_forecast_idx']]['file'],
                self.forecast_data_source['grid_dict']['t_range'],
                datetime.utcfromtimestamp(x_t[3][0]), self.specific_settings['T_goal_in_h']))

        x_t_rel = np.copy(x_t)
        x_t_rel[3] = x_t_rel[3] - self.current_data_t_0
        # log x_t_rel for get_initial_values to access it easily
        self.x_t = x_t_rel

        # Step 2: depending on the reachability direction run the respective algorithm
        if self.specific_settings['direction'] == 'forward':
            self.run_hj_reachability(initial_values=self.get_initial_values(direction="forward"),
                                     t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
                                     dir='forward', x_reach_end=self.get_x_from_full_state(self.x_T), stop_at_x_end=True)
            self.extract_trajectory(x_start=self.get_x_from_full_state(self.x_T), traj_rel_times_vector=None)
        elif self.specific_settings['direction'] == 'backward':
            # Note: no trajectory is extracted as the value function is used for closed-loop control
            self.run_hj_reachability(initial_values=self.get_initial_values(direction="backward"),
                                     t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
                                     dir='backward')
            self.extract_trajectory(x_start=self.get_x_from_full_state(x_t_rel.flatten()), traj_rel_times_vector=None)
            # arrange to forward times by convention for plotting and open-loop control
            self.flip_value_func_to_forward_times()
        elif self.specific_settings['direction'] == 'forward-backward':
            # Step 1: run the set forward to get the earliest possible arrival time
            self.run_hj_reachability(initial_values=self.get_initial_values(direction="forward"),
                                     t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
                                     dir='forward', x_reach_end=self.get_x_from_full_state(self.x_T), stop_at_x_end=True)
            # Step 2: run the set backwards from the earliest arrival time backwards
            _, t_earliest_in_h = self.get_t_earliest_for_target_region()
            print("earliest for target region is ", t_earliest_in_h)
            self.run_hj_reachability(initial_values=self.get_initial_values(direction="backward"),
                                     t_start=x_t_rel[3],
                                     T_max_in_h=t_earliest_in_h + self.specific_settings['fwd_back_buffer_in_h'],
                                     dir='backward')
            self.extract_trajectory(x_start=self.get_x_from_full_state(x_t_rel.flatten()), traj_rel_times_vector=None)
            # arrange to forward times by convention for plotting and open-loop control
            self.flip_value_func_to_forward_times()
        elif self.specific_settings['direction'] == 'multi-reach-back':
            # Step 1: run multi-reachability backwards in time
            self.run_hj_reachability(initial_values=self.get_initial_values(direction="multi-reach-back"),
                                     t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
                                     dir='multi-reach-back')

            # Now just extract it forwards releasing the vehicle at t=0
            def termination_condn(x_target, r, x, t):
                return np.linalg.norm(x_target - x) <= r
            termination_condn = partial(termination_condn, self.x_T, self.problem.x_T_radius)
            self.extract_trajectory(self.get_x_from_full_state(x_t_rel.flatten()),
                                    traj_rel_times_vector=None, termination_condn=termination_condn)
            # arrange to forward times by convention for plotting and open-loop control (aka closed-loop with this)
            self.flip_value_func_to_forward_times()
        else:
            raise ValueError("Direction in controller YAML needs to be one of {backward, forward, forward-backward, multi-reach-back}")

        # check if all_values contains any Nans
        if jnp.isnan(self.all_values).sum() > 0:
            raise ValueError("HJ Planner has NaNs in all values. Something went wrong in solving the PDE.")

    def run_hj_reachability(self, initial_values, t_start, T_max_in_h, dir,
                            x_reach_end=None, stop_at_x_end=False):
        """ Run hj reachability starting with initial_values at t_start for maximum of T_max_in_h
            or until x_reach_end is reached going in the time direction of dir.

            Inputs:
            - initial_values    value function of the initial set, must be same dim as grid.ndim
            - t_start           starting time in seconds relative to the forecast file
            - T_max_in_h        maximum time to run forward reachability for
            - dir               direction for reachability either 'forward' or 'backward'
            - x_reach_end       Optional: target point, must be same dim as grid.ndim (Later can be a region)
            - stop_at_x_end     Optional: bool, stopping the front computation when the target state is reached or not

            Output:             None, everything is set as class variable
            """

        # set the time_scales and offset in the non_dim_dynamics in which the PDE is solved
        self.nondim_dynamics.tau_c = min(T_max_in_h * self.specific_settings['hours_to_hj_solve_timescale'], int(self.current_data_t_T - self.current_data_t_0))
        self.nondim_dynamics.t_0 = t_start

        # set up the non_dimensional time-vector for which to save the value function
        solve_times = np.linspace(0, 1, self.specific_settings['n_time_vector'] + 1)
        # solve_times = t_start + np.linspace(0, T_max_in_h * self.specific_settings['hours_to_hj_solve_timescale'], self.specific_settings['n_time_vector'] + 1)

        if dir == 'backward' or dir == 'multi-reach-back':
            solve_times = np.flip(solve_times, axis=0)
            self.nondim_dynamics.dimensional_dynamics.control_mode = 'min'
            self.nondim_dynamics.dimensional_dynamics.disturbance_mode = 'max'
        elif dir == 'forward':
            self.nondim_dynamics.dimensional_dynamics.control_mode = 'max'
            self.nondim_dynamics.dimensional_dynamics.disturbance_mode = 'min'

        # specific settings for multi-reach-back
        if dir == 'multi-reach-back':
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

        # set variables to stop or not when x_end is in the reachable set
        if stop_at_x_end:
            stop_at_x_init = self.get_non_dim_state(x_reach_end)
        else:
            stop_at_x_init = None

        # create solver settings object
        solver_settings = hj.SolverSettings.with_accuracy(accuracy=self.specific_settings['accuracy'],
                                                          x_init=stop_at_x_init,
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

    def get_t_earliest_for_target_region(self):
        """Helper Function to get the earliest time the forward reachable set overlaps with the target region."""
        # get target_region_mask
        target_region_mask = self.get_initial_values(direction="backward") <= 0

        # iterate forward to get earliest time it's inside
        for idx in range(self.all_values.shape[0]):
            reached = np.logical_and(target_region_mask, self.all_values[idx, ...] <= 0).any()
            if reached:
                break
        # extract earliest relative time of idx
        T_earliest_in_h = (self.reach_times[idx] - self.reach_times[0]) / self.specific_settings['hours_to_hj_solve_timescale']
        if not reached:
            print("Not reached, returning maximum time for the backwards reachability.")
        return reached, T_earliest_in_h

    def extract_trajectory(self, x_start, traj_rel_times_vector=None, termination_condn=None):
        """Backtrack the reachable front to extract a trajectory etc.

        Input Parameters:
        - x_start                   start_point for backtracking must be same dim as grid.ndim
        - traj_rel_times_vector     the times vector for which to extract trajectory points for
                                    in seconds from the start of the reachability computation t=0.
                                    Defaults to self.reach_times.
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

        if self.specific_settings['direction'] in ['backward', 'multi-reach-back', 'forward-backward']:
            self.flip_traj_to_forward_times()

        # log the planned trajectory for later inspection purpose
        # Step 1: concatenate to reduce file size
        times_vec = self.times.reshape(1, -1)
        trajectory = np.concatenate((self.x_traj, np.ones(times_vec.shape), times_vec), axis=0)

        plan_dict = {'traj':trajectory, 'ctrl':self.contr_seq}
        self.planned_trajs.append(plan_dict)

    def update_current_data(self, x_t):
        print("Reachability Planner: Loading new current data.")

        # get the t, lat, lon bounds for sub-setting the data
        t_interval, lat_bnds, lon_bnds = \
            simulation_utils.convert_to_lat_lon_time_bounds(x_t.flatten(), self.x_T,
                                                            deg_around_x0_xT_box=self.specific_settings['deg_around_xt_xT_box'],
                                                            temp_horizon_in_h=self.specific_settings['T_goal_in_h'],
                                                            hours_to_hj_solve_timescale=self.specific_settings['hours_to_hj_solve_timescale'])

        # if it's a forecast we need to update the grid dict
        if 'current_forecast_idx' in self.forecast_data_source:
            self.forecast_data_source['grid_dict'] = self.problem.derive_grid_dict_from_files(self.forecast_data_source)

        # Step 0: check if within the spatial and temporal domain. Otherwise modify.
        lat_bnds = [max(lat_bnds[0], self.forecast_data_source['grid_dict']['y_range'][0]),
                    min(lat_bnds[1], self.forecast_data_source['grid_dict']['y_range'][1])]
        lon_bnds = [max(lon_bnds[0], self.forecast_data_source['grid_dict']['x_range'][0]),
                    min(lon_bnds[1], self.forecast_data_source['grid_dict']['x_range'][1])]

        if self.forecast_data_source['data_source_type'] == 'analytical_function':
            # calculate target shape of the grid
            x_n_res = int((lon_bnds[-1] - lon_bnds[0]) / self.specific_settings['grid_res'][0])
            y_n_res = int((lat_bnds[-1] - lat_bnds[0]) / self.specific_settings['grid_res'][1])
            # get the data subset from analytical field with specific shape
            grids_dict, water_u, water_v = self.forecast_data_source['content'].get_subset_from_analytical_field(
                t_interval,lat_bnds, lon_bnds, spatial_shape=(x_n_res, y_n_res))
            grids_dict['not_plot_land'] = True

        else:
            # get the data subset from the file
            grids_dict, water_u, water_v = simulation_utils.get_current_data_subset(
                t_interval, lat_bnds, lon_bnds,
                data_source=self.forecast_data_source)

            # calculate target shape of the grid
            x_n_res = int((grids_dict['x_grid'][-1] - grids_dict['x_grid'][0])/self.specific_settings['grid_res'][0])
            y_n_res = int((grids_dict['y_grid'][-1] - grids_dict['y_grid'][0])/self.specific_settings['grid_res'][1])

            # do spatial interpolation to the desired resolution to run HJ_reachability
            grids_dict['x_grid'], grids_dict['y_grid'], water_u, water_v = simulation_utils.spatial_interpolation(
                grids_dict, water_u, water_v, target_shape=(y_n_res, x_n_res), kind='linear')

        # set absolute time in UTC Posix time
        self.current_data_t_0 = grids_dict['t_grid'][0]
        # set absolute final time in UTC Posix time
        self.current_data_t_T = grids_dict['t_grid'][-1]

        # feed in the current data to the Platform classes
        # Note: we use a relative time grid (starts with 0 for every file)
        # because otherwise there are errors in the interpolation as jax uses float32
        self.nondim_dynamics.dimensional_dynamics.update_jax_interpolant(
            grids_dict['x_grid'],
            grids_dict['y_grid'],
            np.array([t - self.current_data_t_0 for t in grids_dict['t_grid']]),
            water_u, water_v)

        # initialize the grids and dynamics to solve the PDE with
        self.initialize_hj_grid(grids_dict)
        self.initialize_non_dim_grid()
        # update non_dimensional_dynamics with the new non_dim scaling and offset
        self.nondim_dynamics.characteristic_vec = self.characteristic_vec
        self.nondim_dynamics.offset_vec = self.offset_vec
        # log that we just updated the forecast_file
        self.new_forecast_dicts = False

        # Delete the old caches
        # print("Cache Size: ", hj.solver._solve._cache_size())
        hj.solver._solve._clear_cache()
        xla._xla_callable.cache_clear()

    def get_non_dim_state(self, state):
        """Returns the state transformed from dimensional coordinates to non_dimensional coordinates."""
        return (state.flatten() - self.offset_vec)/self.characteristic_vec

    def initialize_non_dim_grid(self):
        """ Return nondim_grid for the solve."""
        # extract the characteristic scale and offset value for each dimensions
        self.characteristic_vec = self.grid.domain.hi - self.grid.domain.lo
        self.offset_vec = self.grid.domain.lo

        self.nonDimGrid = hj.Grid.nondim_grid_from_dim_grid(
            dim_grid=self.grid, characteristic_vec=self.characteristic_vec, offset_vec=self.offset_vec)

    def flip_traj_to_forward_times(self):
        """ Arrange traj class values to forward for easier access: traj_times, x_traj, contr_seq, distr_seq"""
        # arrange everything forward in time for easier access if we ran it backwards
        if self.times[0] > self.times[-1]:
            self.times = np.flip(self.times, axis=0)
            self.x_traj, self.contr_seq, self.distr_seq = \
                [np.flip(seq, axis=1) for seq in [self.x_traj, self.contr_seq, self.distr_seq]]
        else:
            raise ValueError("Trajectory is already in forward time.")

    def flip_value_func_to_forward_times(self):
        """ Arrange class values to forward for easier access: reach_times and all_values."""
        if self.reach_times[0] > self.reach_times[-1]:
            self.reach_times, self.all_values = [np.flip(seq, axis=0) for seq in [self.reach_times, self.all_values]]
        else:
            raise ValueError("Reachability Values are already in forward time.")

    def get_next_action(self, state, trajectory):
        """Directly getting actions for closed-loop control.
        if forward:     applying the actions from the contr_seq
        if backward:    computing the gradient/action directly from the value function
        """

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
        return np.asarray(u_out.reshape(-1, 1))

    def plot_reachability_snapshot(self, rel_time_in_h,  multi_reachability=False, granularity_in_h=5,
                                   time_to_reach=False, return_ax=False, fig_size_inches=(12, 12)):
        """ Plot the reachable set the planner was computing last. """
        if self.grid.ndim != 2:
            raise ValueError("plot_reachability is currently only implemented for 2D sets")

        # get_initial_value
        initial_values = self.get_initial_values(direction=self.specific_settings['direction'])

        # interpolate
        val_at_t = interp1d(self.reach_times - self.reach_times[0], self.all_values, axis=0, kind='linear')(
            rel_time_in_h * self.specific_settings['hours_to_hj_solve_timescale']).squeeze()

        # If in normal reachability setting
        if not multi_reachability:
            ax = hj.viz._visSet2D(self.grid, val_at_t, level=0, color='black',
                                  colorbar=False, obstacles=None, target_set=initial_values, return_ax=True)
        else:   # multi-reachability
            multi_reach_rel_time = (rel_time_in_h * self.specific_settings['hours_to_hj_solve_timescale'] - self.reach_times[-1])/self.specific_settings['hours_to_hj_solve_timescale']
            non_dim_val_func_levels, abs_time_y_ticks, y_label = self.get_multi_reach_levels(
                granularity_in_h, time_to_reach=time_to_reach, vmin=val_at_t.min(), abs_time_in_h=multi_reach_rel_time)
            # plot with the basic function
            ax = hj.viz._visSet2D(self.grid, val_at_t, level=0, color='black',
                                  colorbar=True, obstacles=None, target_set=initial_values,
                                  val_func_levels=non_dim_val_func_levels, y_label=y_label,
                                  yticklabels=abs_time_y_ticks, return_ax=True)

            ax.scatter(self.x_0[0], self.x_0[1], color='r', marker='o')
            ax.scatter(self.x_T[0], self.x_T[1], color='g', marker='x')
            if self.forecast_data_source['data_source_type'] == 'analytical_function':
                ax.set_title("Multi-Reach at time {} hours".format(
                    self.reach_times[0] + rel_time_in_h * self.specific_settings['hours_to_hj_solve_timescale'] + self.current_data_t_0))
            else:
                ax.set_title("Multi-Reach at time {}".format(datetime.fromtimestamp(
                    self.reach_times[0] + rel_time_in_h * self.specific_settings['hours_to_hj_solve_timescale'] + self.current_data_t_0,
                    tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')))

        # adjust the fig_size
        fig = plt.gcf()
        fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
        if return_ax:
            return ax
        else:
            plt.show()

    def plot_reachability_animation(self, type='gif', multi_reachability=False, granularity_in_h=5, time_to_reach=False):
        """Create an animation of the reachability computation."""
        # Step 0: determine x_init to plot
        if self.specific_settings['direction'] == 'forward':
            x_init = self.x_T
        else:   # backwards
            x_init = self.x_t

        # create the animation
        if not multi_reachability:
            hj.viz.visSet2DAnimation(
                self.grid, self.all_values, (self.reach_times - self.reach_times[0])/self.specific_settings['hours_to_hj_solve_timescale'],
                type=type, x_init=x_init, colorbar=False)
        # Create multi-reachability animation
        else:
            abs_time_in_h_vec = (self.reach_times - self.reach_times[0]) / self.specific_settings['hours_to_hj_solve_timescale']
            non_dim_val_func_levels, abs_time_y_ticks, y_label = self.get_multi_reach_levels(
                granularity_in_h, time_to_reach=time_to_reach, vmin=self.all_values.min(), abs_time_in_h=abs_time_in_h_vec[-1])
            hj.viz.visSet2DAnimation(self.grid, self.all_values, abs_time_in_h_vec,
                                     type=type, x_init=self.x_0[:2], colorbar=True,
                                     val_func_levels=non_dim_val_func_levels, y_label=y_label,
                                     color_yticklabels=abs_time_y_ticks,
                                     filename='2D_multi_reach_animation')


    def vis_Value_func_along_traj(self, figsize=(12,12), return_ax=False, extra_traj=None, time_to_reach=False):
        """Plot the Value function along the most recently planned trajectory."""
        fig, ax = plt.subplots(figsize=figsize)

        if time_to_reach:
            all_values_dimensional = 1 + self.all_values - (self.reach_times / self.reach_times[-1]).reshape(-1, 1, 1)
            all_values = all_values_dimensional * self.specific_settings['T_goal_in_h']
            ylabel = "Earliest-time-to-reach"
        else:
            ylabel = r"$\phi(x_t)$"
            all_values = self.all_values

        reach_times = (self.reach_times - self.reach_times[0]) / self.specific_settings['hours_to_hj_solve_timescale']
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
                                  all_times=self.reach_times / self.specific_settings['hours_to_hj_solve_timescale'],
                                  all_values=all_values, grid=self.grid,
                                  flip_times=False,
                                  ylabel=ylabel)
        if return_ax:
            return ax
        else:
            plt.show()

    @staticmethod
    def get_multi_reach_levels(granularity_in_h, vmin, abs_time_in_h, time_to_reach):
        """Helper function to determine the levels for multi-reachability plotting."""

        n_levels = abs(math.ceil(abs_time_in_h / granularity_in_h)) + 1
        non_dim_val_func_levels = np.linspace(vmin, 0, n_levels)
        abs_time_y_ticks = np.around(np.linspace(abs_time_in_h, 0, n_levels), decimals=1)

        if time_to_reach:
            y_label = 'Fastest Time-to-Target in hours'
            abs_time_y_ticks = -np.flip(abs_time_y_ticks, axis=0)
        else:
            y_label = 'HJ Value Function'

        return non_dim_val_func_levels, abs_time_y_ticks, y_label

    def get_waypoints(self):
        """Returns: a list of waypoints each containing [lon, lat, time]"""
        return np.vstack((self.x_traj, self.times)).T.tolist()

    def set_diss_schema(self):
        # Note: can be done more elegantly by list_indexing =)
        if self.specific_settings['artificial_dissipation_scheme'] == 'local_local':
            self.diss_scheme = hj.artificial_dissipation.local_local_lax_friedrichs
        elif self.specific_settings['artificial_dissipation_scheme'] == 'local':
            self.diss_scheme = hj.artificial_dissipation.local_lax_friedrichs
        elif self.specific_settings['artificial_dissipation_scheme'] == 'global':
            self.diss_scheme = hj.artificial_dissipation.global_lax_friedrichs
        else:
            raise ValueError("artificial_dissipation_scheme is not one of {global, local, local_local}")

