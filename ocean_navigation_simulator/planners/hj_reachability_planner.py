from ocean_navigation_simulator.planners.planner import Planner
import numpy as np
from ocean_navigation_simulator.utils import simulation_utils
import os
from scipy.interpolate import interp1d
import bisect
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
    def __init__(self, problem, gen_settings, specific_settings):
        # initialize Planner superclass
        super().__init__(problem, gen_settings, specific_settings)

        # create a variable that persists across runs of self.plan() to reference the currently reload data
        self.current_data_t_0 = None
        # this is just a variable that persists after planning for plotting/debugging
        self.x_t = None
        # initializes variables needed for planning, they will be filled in the plan method
        self.reach_times, self.all_values, self.times_abs, self.grid, self.diss_scheme = [None]*5
        self.x_traj, self.contr_seq, self.distr_seq = [None]*3
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
    def get_initial_values(self, center):
        """Create the initial value function over the grid must be implemented by specific planner."""
        raise ValueError("get_initial_values must be implemented in the child class")

    # abstractmethod: needs to be implemented for each planner
    def get_x_from_full_state(self, x):
        """Return the x state appropriate for the specific reachability planner."""
        raise ValueError("get_x_start must be implemented in the child class")

    def plan(self, x_t, trajectory=None):
        """Main function where the reachable front is computed."""
        # Step 1: read the relevant subset of data (if it changed)
        if self.new_forecast_file:
            self.update_current_data(x_t=x_t)

        # Check if x_t is in the forecast times and transform to rel_time in seconds
        if x_t[3] < self.current_data_t_0:
            raise ValueError("Current time is before the start of the forecast file. This should not happen.")
        x_t_rel = np.copy(x_t)
        x_t_rel[3] = x_t_rel[3] - self.current_data_t_0

        # Step 2: depending on the reachability direction run the respective algorithm
        if self.specific_settings['direction'] == 'forward':
            self.run_hj_reachability(initial_values=self.get_initial_values(center=x_t_rel),
                                     t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
                                     dir='forward', x_reach_end=self.get_x_from_full_state(self.x_T), stop_at_x_end=True)
            self.extract_trajectory(x_start=self.get_x_from_full_state(self.x_T), traj_rel_times_vector=None)
        elif self.specific_settings['direction'] == 'backward':
            # Note: no trajectory is extracted as the value function is used for closed-loop control
            self.run_hj_reachability(initial_values=self.get_initial_values(center=self.x_T),
                                     t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
                                     dir='backward')
            self.extract_trajectory(x_start=self.get_x_from_full_state(x_t_rel.flatten()), traj_rel_times_vector=None)
            # arrange to forward times by convention for plotting and open-loop control
            self.flip_traj_to_forward_times()
            self.flip_value_func_to_forward_times()
        elif self.specific_settings['direction'] == 'forward-backward':
            # Step 1: run the set forward to get the earliest possible arrival time
            self.run_hj_reachability(initial_values=self.get_initial_values(center=x_t_rel),
                                     t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
                                     dir='forward', x_reach_end=self.get_x_from_full_state(self.x_T), stop_at_x_end=True)
            # Step 2: run the set backwards from the earliest arrival time backwards
            t_earliest_in_h = (self.reach_times[-1] - x_t_rel[3])/3600
            self.run_hj_reachability(initial_values=self.get_initial_values(center=self.x_T),
                                     t_start=x_t_rel[3],
                                     T_max_in_h=t_earliest_in_h[0] + self.specific_settings['fwd_back_buffer_in_h'],
                                     dir='backward')
            self.extract_trajectory(x_start=self.get_x_from_full_state(x_t_rel.flatten()), traj_rel_times_vector=None)
            # arrange to forward times by convention for plotting and open-loop control
            self.flip_traj_to_forward_times()
            self.flip_value_func_to_forward_times()
        else:
            raise ValueError("Direction in controller YAML needs to be one of {backward, forward, forward-backward}")

        # for open_loop control the times vector must be in absolute times
        self.times = self.times + self.current_data_t_0
        # log it for plotting planned trajectory
        self.x_t = x_t

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
        self.nondim_dynamics.tau_c = T_max_in_h * 3600
        self.nondim_dynamics.t_0 = t_start

        # set up the non_dimensional time-vector for which to save the value function
        solve_times = np.linspace(0, 1, self.specific_settings['n_time_vector'] + 1)
        # solve_times = t_start + np.linspace(0, T_max_in_h * 3600, self.specific_settings['n_time_vector'] + 1)

        if dir == 'backward':
            solve_times = np.flip(solve_times, axis=0)
            self.nondim_dynamics.dimensional_dynamics.control_mode = 'min'
        elif dir == 'forward':
            self.nondim_dynamics.dimensional_dynamics.control_mode = 'max'

        # set variables to stop or not when x_end is in the reachable set
        if stop_at_x_end:
            stop_at_x_init = self.get_non_dim_state(x_reach_end)
        else:
            stop_at_x_init = None

        # create solver settings object
        solver_settings = hj.SolverSettings.with_accuracy(accuracy=self.specific_settings['accuracy'],
                                                          x_init=stop_at_x_init,
                                                          artificial_dissipation_scheme=self.diss_scheme)

        # solve the PDE in non_dimensional to get the value function V(s,t)
        non_dim_reach_times, self.all_values = hj.solve(
            solver_settings=solver_settings,
            dynamics=self.nondim_dynamics,
            grid=self.nonDimGrid,
            times=solve_times,
            initial_values=initial_values)

        # scale up the reach_times to be dimensional_times in seconds again
        self.reach_times = non_dim_reach_times * self.nondim_dynamics.tau_c + self.nondim_dynamics.t_0

    def extract_trajectory(self, x_start, traj_rel_times_vector=None):
        """Backtrack the reachable front to extract a trajectory etc.

        Input Parameters:
        - x_start                   start_point for backtracking must be same dim as grid.ndim
        - traj_rel_times_vector     the times vector for which to extract trajectory points for
                                    in seconds relative to the current forecast file.
                                    Defaults to self.reach_times.
        """
        # setting default times vector for the trajectory
        if traj_rel_times_vector is None:
            traj_rel_times_vector = self.reach_times

        self.times, self.x_traj, self.contr_seq, self.distr_seq = \
            self.nondim_dynamics.dimensional_dynamics.backtrack_trajectory(
                grid=self.grid, x_init=x_start, times=self.reach_times, all_values=self.all_values,
                traj_times=traj_rel_times_vector)

    def update_current_data(self, x_t):
        print("Reachability Planner: Loading new current data.")

        t_interval, lat_bnds, lon_bnds = \
            simulation_utils.convert_to_lat_lon_time_bounds(x_t, self.x_T,
                                                            deg_around_x0_xT_box=self.specific_settings['deg_around_xt_xT_box'],
                                                            temp_horizon_in_h=self.specific_settings['T_goal_in_h'])
        grids_dict, water_u, water_v = simulation_utils.get_current_data_subset(self.cur_forecast_file,
                                                                                t_interval, lat_bnds, lon_bnds)


        # set absolute time in Posix time
        self.current_data_t_0 = grids_dict['t_grid'][0]

        # feed in the current data to the Platform classes
        # Note: we use a relative time grid (starts with 0 for every file)
        # because otherwise there are errors in the interpolation as jax uses float32
        self.nondim_dynamics.dimensional_dynamics.update_jax_interpolant(
            grids_dict['x_grid'].data,
            grids_dict['y_grid'].data,
            [t - self.current_data_t_0 for t in grids_dict['t_grid']],
            water_u, water_v)

        # initialize the grids and dynamics to solve the PDE with
        self.initialize_hj_grid(grids_dict)
        self.initialize_non_dim_grid()
        # update non_dimensional_dynamics with the new non_dim scaling and offset
        self.nondim_dynamics.characteristic_vec = self.characteristic_vec
        self.nondim_dynamics.offset_vec = self.offset_vec
        # log that we just updated the forecast_file
        self.new_forecast_file = False

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

    def get_next_action(self, state):
        """Directly getting actions for closed-loop control.
        if forward:     applying the actions from the contr_seq
        if backward:    computing the gradient/action directly from the value function
        """

        if self.specific_settings['direction'] == 'forward':
            u_out = super().get_u_from_vectors(state, ctrl_vec='dir')
        else:
            u_out, _ = self.nondim_dynamics.dimensional_dynamics.get_opt_ctrl_from_values(
                grid=self.grid, x=self.get_x_from_full_state(state),
                time=state[3] - self.current_data_t_0,
                times=self.reach_times, all_values=self.all_values)
        return np.asarray(u_out.reshape(-1, 1))

    def plot_reachability(self):
        """ Plot the reachable set the planner was computing last. """
        if self.grid.ndim != 2:
            raise ValueError("plot_reachability is currently only implemented for 2D sets")
        if self.specific_settings['direction'] == 'forward':
            hj.viz.visSet2DAnimation(
                self.grid, self.all_values, (self.reach_times - self.reach_times[0])/3600,
                type='safari', x_init=self.x_T, colorbar=False)
        else:   # backwards
            hj.viz.visSet2DAnimation(
                self.grid, self.all_values, (self.reach_times - self.reach_times[0])/3600,
                type='safari', x_init=self.x_t, colorbar=False)

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


class HJReach2DPlanner(HJPlannerBase):
    """ Reachability planner for 2D (lat, lon) reachability computation."""

    def get_x_from_full_state(self, x):
        return x[:2]

    def get_dim_dynamical_system(self):
        """Initialize 2D (lat, lon) Platform dynamics in deg/s."""
        # space coefficient is fixed for now as we run in deg/s (same as the simulator)
        space_coeff = 1. / self.gen_settings['conv_m_to_deg']
        return hj.systems.Platform2D_for_sim(u_max=self.dyn_dict['u_max'],
                                             space_coeff=space_coeff, control_mode='min')

    def initialize_hj_grid(self, grids_dict):
        """Initialize the dimensional grid in degrees lat, lon"""
        self.grid = hj.Grid.from_grid_definition_and_initial_values(
            domain=hj.sets.Box(
                lo=np.array([grids_dict['x_grid'][0], grids_dict['y_grid'][0]]),
                hi=np.array([grids_dict['x_grid'][-1], grids_dict['y_grid'][-1]])),
            shape=self.specific_settings['grid_res'])

    def get_initial_values(self, center):
        return hj.shapes.shape_ellipse(grid=self.nonDimGrid,
                                       center=self.get_non_dim_state(self.get_x_from_full_state(center.flatten())),
                                       radii=self.specific_settings['initial_set_radii']/self.characteristic_vec)