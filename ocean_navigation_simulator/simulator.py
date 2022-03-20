import yaml
import glob, imageio, os
import casadi as ca
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
import ocean_navigation_simulator.planners as planners
import ocean_navigation_simulator.steering_controllers as steering_controllers
from ocean_navigation_simulator.utils import plotting_utils, simulation_utils, solar_rad


class OceanNavSimulator:
    """
    This class takes in a simulator and control YAML file for configuration as well as a problem.
    It has access to the forecasted currents for each day and hindcasts over the time period.
    It simulates the closed-loop performance of the running-system in POSIX time.

    sim_config_dict:                Dict specifying the Simulator configuration
                                    see the repos 'configs/simulator.yaml' as example.
                                    If a <string> is provided, assumes python is run locally
                                    from inside the OceanPlatformControl repo and the dict is as yaml under
                                    '/configs/<string>' => this is meant for local testing.
    control_config_dict:            Dict specifying the control configuration
                                    see the repos 'configs/controller.yaml' as example.
                                    If a <string> is provided, see note for sim_config_dict
    problem:                        Problem for the planner (x_0, x_T, t_0, radius, noise)
    #TODO: set up a logging infrastructure with different levels for debugging.
    """

    def __init__(self, sim_config_dict, control_config_dict, problem):
        # read in the sim and control configs from yaml files if run locally
        if isinstance(sim_config_dict, str) and isinstance(control_config_dict, str):
            # get the local project directory (assuming a specific folder structure)
            project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # load simulator sim_settings YAML
            with open(project_dir + '/configs/' + sim_config_dict) as f:
                sim_config_dict = yaml.load(f, Loader=yaml.FullLoader)
                # sim_settings['project_dir'] = project_dir
            # load controller settings YAML
            with open(project_dir + '/configs/' + control_config_dict) as f:
                control_config_dict = yaml.load(f, Loader=yaml.FullLoader)

        # check if update frequencies make sense:
        if control_config_dict['SteeringCtrlConfig']['dt_replanning'] < sim_config_dict['dt']:
            raise ValueError("Simulator dt needs to be smaller than tracking controller replanning dt")

        self.sim_settings = sim_config_dict
        self.control_settings = control_config_dict
        self.problem = problem

        # initialize the GT dynamics (currently HYCOM Hindcasts, later potentially adding noise)
        self.grids_dict = None
        self.F_x_next = None
        self.update_dynamics(self.problem.x_0)

        # create instances for the high-level planner and a pointer to the waypoint tracking function
        # Step 1: initialize high-level planner
        planner_class = getattr(planners, self.control_settings['PlannerConfig']['name'])
        self.high_level_planner = planner_class(self.problem,
                                                specific_settings=self.control_settings['PlannerConfig'][
                                                    'specific_settings'],
                                                conv_m_to_deg=self.sim_settings['conv_m_to_deg'])
        # Step 2: initialize the tracking controller
        if self.control_settings['SteeringCtrlConfig']['name'] != 'None':
            tracking_class = getattr(steering_controllers, self.control_settings['SteeringCtrlConfig']['name'])
            self.steering_controller = tracking_class()
        else:  # do open-loop control using the get_action_function from the high-level planner
            self.steering_controller = self.high_level_planner
            print("Running Open-Loop Control without a steering controller.")

        # initiate some tracking variables
        self.cur_state = np.array(problem.x_0).reshape(4, 1)  # lon, lat, battery level, time
        self.trajectory = self.cur_state
        self.control_traj = np.empty((2, 0), float)

        # termination reason
        self.termination_reason = None

    def update_dynamics(self, x_t):
        """Update symbolic dynamics function for the simulation by sub-setting the relevant set of current data.
        Specifically, the self.F_x_next symbolic function and the self.grid_dict
        Args:
            x_t: current location of agent
        Output:
            F_x_next          cassadi function with (x,u)->(x_next)
        """
        # Step 0: set up time and lat/lon bounds for data sub-setting
        t_upper = min(x_t[3] + self.sim_settings['hours_to_sim_timescale'] * 24 * self.sim_settings["t_horizon_sim"],
                      self.problem.hindcast_data_source['grid_dict']['t_range'][1].timestamp())

        t_interval = [datetime.fromtimestamp(x_t[3], tz=timezone.utc), datetime.fromtimestamp(t_upper, tz=timezone.utc)]
        lon_interval = [x_t[0] - self.sim_settings["deg_around_x_t"], x_t[0] + self.sim_settings["deg_around_x_t"]]
        lat_interval = [x_t[1] - self.sim_settings["deg_around_x_t"], x_t[1] + self.sim_settings["deg_around_x_t"]]

        # Step 1: define variables
        x_sym_1 = ca.MX.sym('x1')  # lon
        x_sym_2 = ca.MX.sym('x2')  # lat
        x_sym_3 = ca.MX.sym('x3')  # battery
        x_sym_4 = ca.MX.sym('t')  # time
        x_sym = ca.vertcat(x_sym_1, x_sym_2, x_sym_3, x_sym_4)

        u_sim_1 = ca.MX.sym('u_1')  # thrust magnitude in [0,1]
        u_sim_2 = ca.MX.sym('u_2')  # header in radians
        u_sym = ca.vertcat(u_sim_1, u_sim_2)

        # Step 2.1: read the relevant subset of data
        self.grids_dict, u_data, v_data = simulation_utils.get_current_data_subset(
            t_interval, lat_interval, lon_interval, self.problem.hindcast_data_source)

        # Step 2.2: get the current interpolation functions
        u_curr_func, v_curr_func = simulation_utils.get_interpolation_func(
            self.grids_dict, u_data, v_data, type=self.sim_settings['int_pol_type'])

        # Step 2.1: relative charging of the battery depends on unix time, lat, lon
        charge = self.problem.dyn_dict['solar_factor'] * solar_rad(x_sym[3], x_sym[1], x_sym[0])

        # Step 3.1: create the x_dot dynamics function
        x_dot_func = ca.Function('f_x_dot', [x_sym, u_sym],
                                 [ca.vertcat((ca.cos(u_sym[1]) * u_sym[0] * self.problem.dyn_dict[
                                     'u_max'] + u_curr_func(ca.vertcat(x_sym[3], x_sym[1], x_sym[0]))) /
                                             self.sim_settings['conv_m_to_deg'],
                                             (ca.sin(u_sym[1]) * u_sym[0] * self.problem.dyn_dict[
                                                 'u_max'] + v_curr_func(
                                                 ca.vertcat(x_sym[3], x_sym[1], x_sym[0]))) / self.sim_settings[
                                                 'conv_m_to_deg'],
                                             charge - self.problem.dyn_dict['energy'] *
                                             (self.problem.dyn_dict['u_max'] * u_sym[0]) ** 3,
                                             1)],
                                 ['x', 'u'], ['x_dot'])

        # Step 3.2: create an integrator out of it
        if self.sim_settings['sim_integration'] == 'rk':
            dae = {'x': x_sym, 'p': u_sym, 'ode': x_dot_func(x_sym, u_sym)}
            integ = ca.integrator('F_int', 'rk', dae, {'tf': self.sim_settings['dt']})
            # Simplify API to (x,u)->(x_next)
            F_x_next = ca.Function('F_x_next', [x_sym, u_sym],
                                   [integ(x0=x_sym, p=u_sym)['xf']], ['x', 'u'], ['x_next'])

        elif self.sim_settings['sim_integration'] == 'ef':
            F_x_next = ca.Function('F_x_next', [x_sym, u_sym],
                                   [x_sym + self.sim_settings['dt'] * x_dot_func(x_sym, u_sym)],
                                   ['x', 'u'], ['x_next'])
        else:
            raise ValueError('sim_integration: only RK4 (rk) and forward euler (ef) implemented')

        # set the class variable
        self.F_x_next = F_x_next

    def run(self, T_in_h=None):
        """Main Loop of the simulator including replanning etc.
        The Loop runs with step-size dt for time T_in_h or until the goal is reached or terminated because stranded.
        Returns string of the termination reason for the simulator.
        'goal_reached', 'stranded', 'T_max_reached', 'outside_spatial_domain_of_hindcasts', 'need_new_temporal_data'

        # TODO: for now there is one main loop, at some point we want to have parallel processes/workers
        running the high-level planner, tracking controller, simulator at their respective frequencies.
        This has 2 advantages: 1) the can run on actual frequencies and not limited by for loop, 2) speed up!
        """
        if self.problem.plan_on_gt:
            self.high_level_planner.update_forecast_dicts(self.problem.hindcast_data_source)
        else:
            self.high_level_planner.update_forecast_dicts(self.problem.forecast_data_source)

        # tracking variables
        end_sim, self.termination_reason = self.check_termination(T_in_h)
        next_planner_update = 0.
        next_steering_controller_update = 0.

        while not end_sim:
            # check to update current data in the dynamics
            self.check_dynamics_update()
            # Loop 1: update forecast files if new one is available
            if not self.problem.plan_on_gt and 'current_forecast_idx' in self.problem.forecast_data_source:
                next_fmrc_idx = self.problem.forecast_data_source['current_forecast_idx'] + 1
                if self.cur_state[3] >= self.problem.forecast_data_source['content'][next_fmrc_idx]['t_range'][0].timestamp():
                    # update current_forecast_idx by + 1
                    self.problem.forecast_data_source['current_forecast_idx'] = next_fmrc_idx
                    # update the grid dict inside the high_level_planner
                    self.high_level_planner.update_forecast_dicts(self.problem.forecast_data_source)
                    if next_fmrc_idx == len(self.problem.forecast_data_source['content']):
                        raise ValueError("Not enough forecasts in the folder to complete simulation.")
                    # trigger high-level planner re-planning
                    next_planner_update = self.cur_state[3]

            # Loop 2: replan with high-level planner
            if self.cur_state[3] >= next_planner_update:
                self.high_level_planner.plan(self.cur_state, trajectory=None)
                self.steering_controller.set_waypoints(self.high_level_planner.get_waypoints())
                self.steering_controller.replan(self.cur_state)
                # set new updating times
                next_planner_update = self.cur_state[3] \
                                      + self.control_settings['PlannerConfig']['dt_replanning']
                next_steering_controller_update = self.cur_state[3] \
                                                  + self.control_settings['SteeringCtrlConfig']['dt_replanning']
                print("High-level planner & tracking controller replanned")

            # Loop 3: replanning of Waypoint tracking
            if self.cur_state[3] >= next_steering_controller_update:
                self.steering_controller.replan(self.cur_state)
                next_steering_controller_update = self.cur_state[3] \
                                                  + self.control_settings['SteeringCtrlConfig']['dt_replanning']
                print("Steering controller replanned")

            # simulator step
            self.run_step()
            print("Sim step")

            # check termination
            end_sim, self.termination_reason = self.check_termination(T_in_h)
        return self.termination_reason

    def check_dynamics_update(self):
        """ Helper function for main loop to check if we need to load new current data into the dynamics."""
        x_low, x_high = self.grids_dict["x_grid"][0], self.grids_dict["x_grid"][-1]
        y_low, y_high = self.grids_dict["y_grid"][0], self.grids_dict["y_grid"][-1]
        t_low, t_high = self.grids_dict["t_grid"][0], self.grids_dict["t_grid"][-1]
        # check based on space and time of the currently loaded current data if new data needs to be loaded
        if not (x_low < self.cur_state[0] < x_high) \
                or not (y_low < self.cur_state[1] < y_high) \
                or not (t_low < self.cur_state[3] < t_high):
            print("Updating simulator dynamics with new current data.")
            self.update_dynamics(self.cur_state.flatten())

    def run_step(self):
        """Run the simulator for one dt step"""

        u = self.thrust_check(self.steering_controller.get_next_action(self.cur_state))

        # update simulator states
        self.control_traj = np.append(self.control_traj, u, axis=1)
        self.cur_state = self.battery_check(np.array(self.F_x_next(self.cur_state, u)).astype('float64'))
        # add new state to trajectory
        self.trajectory = np.hstack((self.trajectory, self.cur_state))

    def thrust_check(self, u_planner):
        """If the thrust would use more energy than available adjust accordingly."""

        charge = self.problem.dyn_dict['solar_factor'] \
                 * solar_rad(self.cur_state[3], self.cur_state[1], self.cur_state[0])

        delta_charge = charge - \
                       self.problem.dyn_dict['energy'] * (self.problem.dyn_dict['u_max'] * u_planner[0]) ** 3

        next_charge = self.cur_state[2] + delta_charge * self.sim_settings['dt']

        # if smaller than 0.: change the thrust accordingly
        if next_charge < 0.:
            energy_available = self.cur_state[2]
            u_planner[0] = ((charge - energy_available / self.sim_settings['dt']) / \
                            (self.problem.dyn_dict['energy'] * self.problem.dyn_dict['u_max'] ** 3)) ** (1. / 3)
            return u_planner
        else:
            return u_planner

    @staticmethod
    def battery_check(cur_state):
        """Prevents battery level to go above 1."""
        if cur_state[2] > 1.:
            cur_state[2] = 1.
        return cur_state

    def check_termination(self, T_in_h):
        """Helper function returning boolean and termination string.
        If False: simulation is continued, if True it ends
        Termination string is any of:
        'goal_reached', 'stranded', 'T_max_reached', 'outside_spatial_domain_of_hindcasts', 'need_new_temporal_data. """

        t_sim_run = (self.cur_state[3] - self.problem.x_0[3])

        # Step 1: check for major termination reasons
        if self.reached_goal():
            print("Sim terminate because goal reached.")
            return True, "goal_reached"
        if (t_sim_run / self.sim_settings['hours_to_sim_timescale']) > T_in_h:
            print("Sim terminate because T_in_h is over")
            return True, "T_max_reached"
        if self.platform_is_on_land():
            print("Sim terminated because platform stranded on land")
            return True, "stranded"

        # Step 2: check current data files to see if we can continue
        # Step 2.1: check if simulation went over the existing data to load new data
        if self.cur_state[3] > self.problem.hindcast_data_source['grid_dict']['t_range'][-1].timestamp():
            print("Sim paused because hindcast fieldset time is over")
            return True, "need_new_temporal_data"
        # check if we're going outside of the spatial data available.
        # Note: this assumes that all hindcast files have the same spatial data range
        if (not self.problem.hindcast_data_source['grid_dict']['x_range'][0] <= self.cur_state[0] <=
                self.problem.hindcast_data_source['grid_dict']['x_range'][-1]) \
                or (not self.problem.hindcast_data_source['grid_dict']['y_range'][0] <= self.cur_state[1] <=
                        self.problem.hindcast_data_source['grid_dict']['y_range'][-1]):
            print("Sim terminate because state went out of the spatial domains of the hindcast files.")
            return True, "outside_spatial_domain_of_hindcasts"

        return False, "not_terminated"

    def reached_goal(self):
        """Returns whether we have reached the target region."""
        distance_to_center = np.sqrt(
            (self.cur_state[0][0] - self.problem.x_T[0]) ** 2 + (self.cur_state[1][0] - self.problem.x_T[1]) ** 2)
        reached = distance_to_center < self.problem.x_T_radius
        return reached

    def platform_is_on_land(self):
        """Returns True/False if the closest grid_point to the self.cur_state is on_land."""
        # get idx of closest grid-points
        x_idx = (np.abs(self.grids_dict['x_grid'] - self.cur_state[0][0])).argmin()
        y_idx = (np.abs(self.grids_dict['y_grid'] - self.cur_state[1][0])).argmin()
        # Note: the spatial_land_mask is an array with [Y, X]
        return self.grids_dict['spatial_land_mask'][y_idx, x_idx]

    def plot_land_mask(self):
        """Plot the land mask of the current data-subset."""
        plotting_utils.plot_land_mask(self.grids_dict)

    def create_animation_with_plans(self, plan_temporal_stride=5, alpha=0.8, with_control=True,
                                    plan_traj_color='grey', plan_u_color='orange',
                                    html_render=None, vid_file_name=None, deg_around_x0_xT_box=0.5,
                                    sim_temporal_stride=1, time_interval_between_pics=200, fps=10,
                                    linewidth=2, marker=None, linestyle='-', add_ax_func=None):
        """Function to animate the simulated trajectory with planned currents."""
        if not hasattr(self.high_level_planner, 'planned_trajs'):
            raise ValueError("Selected High-Level planner does not log plans in planned_trajs.")

        # Step 1: create add_ax function which plots the most recent plan on top
        def add_current_planned_traj(ax, time, planned_trajs_dicts, temporal_stride, alpha, with_control,
                                     plan_traj_color, plan_u_color):
            # Step 0: check which is the current plan that is followed
            if time >= planned_trajs_dicts[0]['traj'][3, 0]:
                plan_idx = 0
                for idx, plan_dict in enumerate(planned_trajs_dicts[1:]):
                    if time >= plan_dict['traj'][3, 0]:
                        plan_idx = idx + 1
                    else:
                        break
            # % Step 1: plot that plan
            # check if last element is included if not add it
            planning_dict = planned_trajs_dicts[plan_idx]
            if not np.arange(planning_dict['traj'].shape[1])[::temporal_stride][-1] == planning_dict['traj'].shape[
                1] - 1:
                x_traj = np.concatenate(
                    (planning_dict['traj'][:, ::temporal_stride], planning_dict['traj'][:, -1].reshape(-1, 1)),
                    axis=1)
            else:
                x_traj = planning_dict['traj'][:, ::temporal_stride]
            ax.plot(x_traj[0, :], x_traj[1, :], '--', c=plan_traj_color, linewidth=2, alpha=alpha, label="current_plan")

            # plot control if supplied
            if with_control:
                # calculate u and v direction
                u_vec = planning_dict['ctrl'][0, ::temporal_stride] * np.cos(
                    planning_dict['ctrl'][1, ::temporal_stride])
                v_vec = planning_dict['ctrl'][0, ::temporal_stride] * np.sin(
                    planning_dict['ctrl'][1, ::temporal_stride])
                ax.quiver(x_traj[0, :-1], x_traj[1, :-1], u_vec, v_vec, color=plan_u_color, scale=10, alpha=alpha,
                          label="planned_ctrl")

            # add any additional axis functions
            add_ax_func(ax, time)
            plt.legend(loc='upper right')

        # Step 2: make a partial out of the function
        add_plan_partial = partial(add_current_planned_traj,
                                   planned_trajs_dicts=self.high_level_planner.planned_trajs,
                                   temporal_stride=plan_temporal_stride, alpha=alpha, with_control=with_control,
                                   plan_traj_color=plan_traj_color, plan_u_color=plan_u_color)

        self.create_animation(html_render=html_render, vid_file_name=vid_file_name,
                              deg_around_x0_xT_box=deg_around_x0_xT_box, temporal_stride=sim_temporal_stride,
                              time_interval_between_pics=time_interval_between_pics, fps=fps,
                              linewidth=linewidth, marker=marker, linestyle=linestyle, add_ax_func=add_plan_partial)

    def create_animation(self, html_render=None, vid_file_name=None,
                        deg_around_x0_xT_box=0.5, temporal_stride=1, time_interval_between_pics=200, fps=10,
                        linewidth=1.5, marker='x', linestyle='--', add_ax_func=None):
        """Function to create an animation of the simulation run."""

        # check if last element is included if not add it
        if not np.arange(self.trajectory.shape[1])[::temporal_stride][-1] == self.trajectory.shape[1] - 1:
            x_traj = np.concatenate((self.trajectory[:, ::temporal_stride], self.trajectory[:, -1].reshape(-1, 1)),
                                    axis=1)
        else:
            x_traj = self.trajectory[:, ::temporal_stride]

        plotting_utils.plot_2D_traj_animation(
            x_T=self.problem.x_T,
            x_T_radius=self.problem.x_T_radius,
            traj_full=x_traj,
            control_traj=self.control_traj[:, ::temporal_stride],
            file_dicts=self.problem.hindcast_data_source,
            u_max=self.problem.dyn_dict['u_max'],
            deg_around_x0_xT_box=deg_around_x0_xT_box,
            html_render=html_render, filename=vid_file_name,
            time_interval_between_pics=time_interval_between_pics,
            fps=fps,
            linewidth=linewidth, marker=marker, linestyle=linestyle,
            add_ax_func_ext=add_ax_func
        )

    def plot_trajectory(self, plotting_type='2D', time_for_currents=None,
                        deg_around_x0_xT_box=0.5, temporal_stride=1, return_ax=False):
        """ Captures the whole trajectory - energy, position, etc over time"""
        # process the time that is put in the for the 2D static plots
        if time_for_currents is None:
            time_for_currents = self.problem.x_0[3]
        else:
            time_for_currents = time_for_currents.timestamp()

        # check if last element is included if not add it
        if not np.arange(self.trajectory.shape[1])[::temporal_stride][-1] == self.trajectory.shape[1] - 1:
            x_traj = np.concatenate((self.trajectory[:, ::temporal_stride], self.trajectory[:, -1].reshape(-1, 1)),
                                    axis=1)
        else:
            x_traj = self.trajectory[:, ::temporal_stride]

        if plotting_type == '2D':
            ax = plotting_utils.plot_2D_traj(x_traj[:2, :], title="Simulator Trajectory", return_ax=return_ax)
            return ax

        elif plotting_type == '2D_w_currents':
            print("Plotting 2D trajectory with the true currents at time_for_currents/t_0")
            ax = plotting_utils.plot_2D_traj_over_currents(x_traj[:2, :],
                                                           deg_around_x0_xT_box=deg_around_x0_xT_box,
                                                           time=time_for_currents,
                                                           x_T=self.problem.x_T[:2], x_T_radius=self.problem.x_T_radius,
                                                           file_dicts=self.problem.hindcast_data_source,
                                                           return_ax=return_ax)
            return ax

        elif plotting_type == '2D_w_currents_w_controls':
            print("Plotting 2D trajectory with the true currents at time_for_currents/t_0")
            ax = plotting_utils.plot_2D_traj_over_currents(x_traj[:2, :],
                                                           deg_around_x0_xT_box=deg_around_x0_xT_box,
                                                           time=time_for_currents,
                                                           x_T=self.problem.x_T[:2], x_T_radius=self.problem.x_T_radius,
                                                           file_dicts=self.problem.hindcast_data_source,
                                                           ctrl_seq=self.control_traj[:, ::temporal_stride],
                                                           u_max=self.problem.dyn_dict['u_max'],
                                                           return_ax=return_ax)
            return ax

        elif plotting_type == 'ctrl':
            plotting_utils.plot_opt_ctrl(x_traj[3, :-1], self.control_traj[:, ::temporal_stride],
                                         title="Simulator Control Trajectory")

        elif plotting_type == 'battery':
            fig, ax = plt.subplots(1, 1)
            # some stuff for flexible date axis
            locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            # plot
            dates = [datetime.fromtimestamp(posix, tz=timezone.utc) for posix in x_traj[3, :]]
            ax.plot(dates, x_traj[2, :])
            # set axis and stuff
            ax.set_title('Battery charge over time')
            ax.set_ylim(0., 1.1)
            plt.xlabel('time in h')
            plt.ylabel('Battery Charging level [0,1]')
            plt.show()
            return

    def plot_2D_traj_w_plans(self, plan_idx=None, underlying_plot_type='2D', with_control=False, alpha=0.8,
                             plan_temporal_stride=2, true_temporal_stride=1):
        """Function to plot the 2D true Trajectory with the planned trajectories on the way."""""
        # Step 1: plot true trajctory leveraging existing functions
        ax = self.plot_trajectory(plotting_type=underlying_plot_type, return_ax=True, temporal_stride=true_temporal_stride)
        # check if the high-level planner logs the planned trajectories
        if hasattr(self.high_level_planner, 'planned_trajs'):
            # create a colormap
            colors = plt.cm.RdPu(np.linspace(0, 1, len(self.high_level_planner.planned_trajs)))
            # now plot all of them
            if plan_idx is None:
                to_slice = slice(None)
            else:
                to_slice = slice(plan_idx, plan_idx + 1)
            for idx, planning_dict in enumerate(self.high_level_planner.planned_trajs[to_slice]):
                # check if last element is included if not add it
                if not np.arange(planning_dict['traj'].shape[1])[::plan_temporal_stride][-1] == planning_dict['traj'].shape[1] - 1:
                    x_traj = np.concatenate(
                        (planning_dict['traj'][:, ::plan_temporal_stride], planning_dict['traj'][:, -1].reshape(-1, 1)),
                        axis=1)
                else:
                    x_traj = planning_dict['traj'][:, ::plan_temporal_stride]
                ax.plot(x_traj[0, :], x_traj[1, :], '--', c=colors[idx], linewidth=2, alpha=alpha)
                # plot control if supplied
                if with_control:
                    # calculate u and v direction
                    u_vec = planning_dict['ctrl'][0, ::plan_temporal_stride] * np.cos(
                        planning_dict['ctrl'][1, ::plan_temporal_stride])
                    v_vec = planning_dict['ctrl'][0, ::plan_temporal_stride] * np.sin(
                        planning_dict['ctrl'][1, ::plan_temporal_stride])
                    ax.quiver(x_traj[0, :-1], x_traj[1, :-1], u_vec, v_vec, color=colors[idx], scale=15, alpha=alpha)
            plt.legend(loc='upper right')
            plt.show()
        else:
            raise ValueError("Selected High-Level planner does not log plans in planned_trajs.")