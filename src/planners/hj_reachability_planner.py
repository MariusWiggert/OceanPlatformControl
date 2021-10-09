from src.planners.planner import Planner
import casadi as ca
import numpy as np
from src.utils import plotting_utils, simulation_utils, hycom_utils
from scipy.interpolate import interp1d
import bisect
import sys
# TODO: do this in a cleaner way with hj_reachability as a submodule of the repo
sys.path.extend(['/Volumes/Data/2_Work/2_Graduate_Research/1_Seaweed/Reachability_Code/hj_reachability_c3'])
import hj_reachability as hj


class HJReach2DPlanner(Planner):
    """Planner based on Hamilton Jacobi Reachability using backwards reachability from the target.
    For details see: "A future for intelligent autonomous ocean observing systems" P.F.J. Lermusiaux

    Note: currently the PDE runs in deg/s and non-dimensionalization is NOT used.

    Attributes required in the specific_settings dict
            direction: string of {'forward', 'backward'}
                If forward or backwards reachability should be run.
            T_goal_in_h: float
                If backward: the final time to be used for back propagating the reachable set.
                If forward:  the time to be used to forward propagate the reachable set.
            initial_set_radius: float
                The radius of the circular initial set of the value function in degree lat, lon.
            n_time_vector: int
                The number of elements in the time vector which determines how granular the value function is saved.
            grid_res: tuple (int, int)
                The granularity of the 2D grid that is used.
            deg_around_xt_xT_box: float
                how much degree lon, lat around the x_t, x_T we cut out to plan on.
            accuracy: string one of {"low", "medium", "high", "very_high"}
                Determines the accuracy with which the PDE solver is run.
            artificial_dissipation_scheme: string one of {"global", "local", "local_local"}
                Determines which dissipation term is used in the Lax-Friedrichs Time-integration approximation.

    See Planner class for the rest of the attributes.
    """

    def __init__(self, problem, gen_settings, specific_settings):
        # initialize superclass
        super().__init__(problem, gen_settings, specific_settings)
        # space coefficient is fixed for now as we run in deg/s (same as the simulator)
        space_coeff = 1. / self.gen_settings['conv_m_to_deg']

        # create a variable that persists across runs of self.plan() to net reload data
        self.current_data_t_0 = None
        # this is just a variable that persists after planning for plotting/debugging
        self.x_t = None
        # initializes variables needed for planning, they will be filled in the plan method
        self.reach_times, self.all_values, self.times_abs, self.grid, self.diss_scheme = [None]*5
        self.x_traj, self.contr_seq, self.distr_seq = [None]*3
        self.set_diss_schema()
        # initialize systems for max/min respectiveley (to avoid recompilation in forward-backwards)
        self.Plat_fwd = hj.systems.Platform2D_for_sim(u_max=self.dyn_dict['u_max'],
                                                      space_coeff=space_coeff, control_mode='max')
        self.Plat_back = hj.systems.Platform2D_for_sim(u_max=self.dyn_dict['u_max'],
                                                       space_coeff=space_coeff, control_mode='min')

    def update_current_data(self, x_t):
        print("Reachability Planner: Loading new current data.")
        grids_dict, water_u, water_v = simulation_utils.get_current_data_subset(
            nc_file=self.cur_forecast_file,
            x_0=x_t, x_T=self.x_T,
            deg_around_x0_xT_box=self.specific_settings['deg_around_xt_xT_box'],
            temporal_stride=self.gen_settings["temporal_stride"],
            temp_horizon_in_h=self.specific_settings['T_goal_in_h'])

        # set absolute time in Posix time
        self.current_data_t_0 = grids_dict['t_grid'][0]

        # feed in the current data to the Platform classes
        self.Plat_fwd.update_jax_interpolant(
            grids_dict['x_grid'].data,
            grids_dict['y_grid'].data,
            [t - self.current_data_t_0 for t in grids_dict['t_grid']],
            water_u, water_v)
        self.Plat_back.update_jax_interpolant(
            grids_dict['x_grid'].data,
            grids_dict['y_grid'].data,
            [t - self.current_data_t_0 for t in grids_dict['t_grid']],
            water_u, water_v)

        # initialize grid to solve PDE on
        self.grid = hj.Grid.from_grid_definition_and_initial_values(
            domain=hj.sets.Box(
                lo=np.array([grids_dict['x_grid'][0], grids_dict['y_grid'][0]]),
                hi=np.array([grids_dict['x_grid'][-1], grids_dict['y_grid'][-1]])),
            shape=self.specific_settings['grid_res'])
        self.new_forecast_file = False

    def run_forward(self, x_t, T_max_in_h, extract_traj=True):
        """ Run forward reachability starting from x_t for maximum of T_max_in_h. """
        # set start and end
        x_reach_start = x_t[:2].flatten()
        x_reach_end = self.x_T

        rel_times = x_t[3] + np.linspace(0, T_max_in_h * 3600, self.specific_settings['n_time_vector'] + 1)

        # initialize values and setting
        initial_values = hj.shapes.shape_sphere(grid=self.grid, center=x_reach_start,
                                                radius=self.specific_settings['initial_set_radius'])
        solver_settings = hj.SolverSettings.with_accuracy(accuracy=self.specific_settings['accuracy'],
                                                          x_init=x_reach_end,
                                                          artificial_dissipation_scheme=self.diss_scheme
                                                          )
        # run the HJ reachability solver
        self.reach_times, self.all_values = hj.solve(solver_settings,
                                                     self.Plat_fwd, self.grid,
                                                     rel_times, initial_values)
        # backtrack the reachable front to extract trajectory etc.
        if extract_traj:
            _, self.x_traj, self.contr_seq, self.distr_seq = self.Plat_fwd.backtrack_trajectory(
                self.grid, x_reach_end, self.reach_times, self.all_values, solver_settings)

    def run_backward(self, x_t, T_start_in_h, stop_at_x_t=False):
        """ Run backwards reachability starting from x_T at T_start_in_h. """
        # set start
        x_reach_start = self.x_T

        # abs time vector in posix time
        rel_times = x_t[3] + np.linspace(T_start_in_h * 3600, 0, self.specific_settings['n_time_vector'] + 1)

        # initialize values and setting
        initial_values = hj.shapes.shape_sphere(grid=self.grid, center=x_reach_start,
                                                radius=self.specific_settings['initial_set_radius'])
        if stop_at_x_t:
            solver_settings = hj.SolverSettings.with_accuracy(accuracy=self.specific_settings['accuracy'],
                                                              x_init=x_t[:2].flatten(),
                                                              artificial_dissipation_scheme=self.diss_scheme
                                                              )
        else:
            solver_settings = hj.SolverSettings.with_accuracy(accuracy=self.specific_settings['accuracy'],
                                                              artificial_dissipation_scheme=self.diss_scheme
                                                              )
        # run the HJ reachability solver
        self.reach_times, self.all_values = hj.solve(solver_settings,
                                                     self.Plat_back, self.grid,
                                                     rel_times, initial_values)

        # backtrack the reachable front to extract trajectory etc.
        _, self.x_traj, self.contr_seq, self.distr_seq = self.Plat_back.backtrack_trajectory(
            self.grid, x_t[:2].flatten(), self.reach_times, self.all_values, solver_settings)

        # arrange everything forward in time for easier access
        if self.specific_settings['direction'] == 'backward':
            self.x_traj, self.contr_seq, self.distr_seq = \
                [np.flip(seq, axis=1) for seq in [self.x_traj, self.contr_seq, self.distr_seq]]
            self.reach_times, self.all_values = [np.flip(seq, axis=0) for seq in [self.reach_times, self.all_values]]

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
            self.run_forward(x_t=x_t_rel, T_max_in_h=self.specific_settings['T_goal_in_h'])
        elif self.specific_settings['direction'] == 'backward':
            self.run_backward(x_t=x_t_rel, T_start_in_h=self.specific_settings['T_goal_in_h'], stop_at_x_t=False)
        elif self.specific_settings['direction'] == 'forward-backward':
            # Step 1: run the set forward to get the earliest arrival time
            self.run_forward(x_t=x_t_rel, T_max_in_h=self.specific_settings['T_goal_in_h'], extract_traj=False)
            # Step 2: run the set backwards from the earliest arrival time backwards
            t_earliest_in_h = (self.reach_times[-1] - x_t_rel[3])/3600
            self.run_backward(x_t=x_t_rel,
                              T_start_in_h=t_earliest_in_h[0] + self.specific_settings['fwd_back_buffer_in_h'],
                              stop_at_x_t=False)
        else:
            raise ValueError("Direction in controller YAML needs to be one of {backward, forward, forward-backward}")

        # for open_loop control the times vector must be in absolute times
        self.times = self.reach_times + self.current_data_t_0
        # log it for plotting planned trajectory
        self.x_t = x_t

    def get_next_action(self, state):
        """Directly getting actions for closed-loop control.
        if forward:     applying the actions from the contr_seq
        if backward:    computing the gradient/action directly from the value function
        """

        if self.specific_settings['direction'] == 'forward':
            u_out = super().get_u_from_vectors(state, ctrl_vec='dir')
        else:
            u_out = self.get_opt_ctrl_from_values(state)
        return u_out

    def get_opt_ctrl_from_values(self, state):
        # TODO: this should be in the dynamics baseclass for multiple methods to access.
        # Step 0: interpolate the value function for the specific time along the time axis (0)
        # Note: this can probably be done more efficiently e.g. by initializing the function once?
        val_at_t = interp1d(self.times, self.all_values, axis=0, kind='linear')(state[3]).squeeze()

        # Step 1: get center approximation of gradient at current point x
        left_grad_values, right_grad_values = self.grid.upwind_grad_values(
            hj.finite_differences.upwind_first.WENO3, values=val_at_t)

        grad_at_x_cur = self.grid.interpolate(values=0.5 * (left_grad_values + right_grad_values),
                                              state=state[:2].flatten())

        # Step 2: get u_opt and d_opt
        u_opt, _ = self.Plat_back.optimal_control_and_disturbance(
            state=state[:2].flatten(), time=state[3], grad_value=grad_at_x_cur)

        # Step 3: return
        return np.asarray(u_opt.reshape(-1, 1))

    def plot_reachability(self):
        """ Plot the reachable set the planner was computing last. """
        if self.specific_settings['direction'] == 'forward':
            hj.viz.visSet2DAnimation(
                self.grid, self.all_values, (self.reach_times - self.reach_times[0])/3600,
                type='safari', x_init=self.x_T, colorbar=False)
        else:   # backwards
            hj.viz.visSet2DAnimation(
                self.grid, np.flip(self.all_values, 0), (np.flip(self.reach_times, 0) - self.reach_times[-1]) / 3600,
                type='safari', x_init=self.x_t, colorbar=False)

    def get_waypoints(self):
        """Returns: a list of waypoints each containing [lon, lat, time]"""
        return np.vstack((self.x_traj, self.reach_times)).T.tolist()

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
