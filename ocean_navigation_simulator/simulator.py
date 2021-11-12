import yaml
import glob, imageio, os
import casadi as ca
import numpy as np
import ocean_navigation_simulator.planners as planners
import ocean_navigation_simulator.steering_controllers as steering_controllers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from ocean_navigation_simulator.utils import plotting_utils, simulation_utils
import bisect
from ocean_navigation_simulator.solar_radiation.solar_rad import solar_rad


class OceanNavSimulator:
    """
    This class takes in a simulator and control YAML file for configuration as well as a problem.
    It has access to the forecasted currents for each day and hindcasts over the time period.
    It simulates the closed-loop performance of the running-system in POSIX time.

    sim_config:                     Simulator YAML configuration file
    control_config:                 Controller YAML configuration file
    problem:                        Problem for the planner (x_0, x_T, t_0, radius, noise)
    project_dir (optional):         Directory of the project for access to config YAMLs and data
    #TODO: set up a logging infrastructure with different levels for debugging.
    """

    def __init__(self, sim_config, control_config, problem, project_dir=None):
        # get project directory
        if project_dir is None:
            self.project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        else:
            self.project_dir = project_dir
        # load simulator sim_settings YAML
        with open(self.project_dir + '/configs/' + sim_config) as f:
            sim_settings = yaml.load(f, Loader=yaml.FullLoader)
            sim_settings['project_dir'] = project_dir

        # load controller settings YAML
        with open(self.project_dir + '/configs/' + control_config) as f:
            control_settings = yaml.load(f, Loader=yaml.FullLoader)

        # check if update frequencies make sense:
        if control_settings['waypoint_tracking']['dt_replanning'] < sim_settings['dt']:
            raise ValueError("Simulator dt needs to be smaller than tracking controller replanning dt")

        self.sim_settings = sim_settings
        self.control_settings = control_settings
        self.problem = problem

        # initialize the GT dynamics (currently HYCOM Hindcasts, later potentially adding noise)
        self.F_x_next = self.initialize_dynamics()

        # create instances for the high-level planner and a pointer to the waypoint tracking function
        # Step 1: initialize high-level planner
        planner_class = getattr(planners, self.control_settings['planner']['name'])
        self.high_level_planner = planner_class(self.problem,
                                                gen_settings=self.control_settings['planner']['gen_settings'],
                                                specific_settings=self.control_settings['planner']['specific_settings'])
        # Step 2: initialize the tracking controller
        if self.control_settings['waypoint_tracking']['name'] != 'None':
            tracking_class = getattr(steering_controllers, self.control_settings['waypoint_tracking']['name'])
            self.wypt_tracking_contr = tracking_class()
        else:  # do open-loop control using the get_action_function from the high-level planner
            self.wypt_tracking_contr = self.high_level_planner
            print("Running Open-Loop Control without a tracking controller.")

        # initiate some tracking variables
        self.cur_state = np.array(problem.x_0).reshape(4, 1)  # lon, lat, battery level, time
        self.trajectory = self.cur_state
        self.control_traj = np.empty((2, 0), float)

    def initialize_dynamics(self):
        """Initialize symbolic dynamics function for simulation
        Output:
            F_x_next          cassadi function with (x,u)->(x_next)
        """
        # Step 1: define variables
        x_sym_1 = ca.MX.sym('x1')  # lon
        x_sym_2 = ca.MX.sym('x2')  # lat
        x_sym_3 = ca.MX.sym('x3')  # battery
        x_sym_4 = ca.MX.sym('t')  # time
        x_sym = ca.vertcat(x_sym_1, x_sym_2, x_sym_3, x_sym_4)

        u_sim_1 = ca.MX.sym('u_1')  # thrust magnitude in [0,1]
        u_sim_2 = ca.MX.sym('u_2')  # header in radians
        u_sym = ca.vertcat(u_sim_1, u_sim_2)

        # Step 2: read the relevant subset of data
        self.grids_dict, u_data, v_data = \
            simulation_utils.get_current_data_subset(self.problem.hindcast_file,
                                                     self.problem.x_0, self.problem.x_T,
                                                     self.sim_settings['deg_around_x0_xT_box'],
                                                     self.problem.fixed_time,
                                                     self.sim_settings['temporal_stride'])

        # Step 2: get the current interpolation functions
        u_curr_func, v_curr_func = simulation_utils.get_interpolation_func(
            self.grids_dict, u_data, v_data, self.sim_settings['int_pol_type'], self.problem.fixed_time)

        # Step 2.1: relative charging of the battery depends on unix time, lat, lon
        charge = self.problem.dyn_dict['solar_factor'] * solar_rad(x_sym[3], x_sym[1], x_sym[0])

        # Step 3: create the x_dot dynamics function
        if self.problem.fixed_time is None:  # time varying current
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
        else:  # fixed current
            x_dot_func = ca.Function('f_x_dot', [x_sym, u_sym],
                                     [ca.vertcat((ca.cos(u_sym[1]) * u_sym[0] * self.problem.dyn_dict['u_max'] /
                                                  + u_curr_func(ca.vertcat(x_sym[1], x_sym[0]))) / self.sim_settings[
                                                     'conv_m_to_deg'],
                                                 (ca.sin(u_sym[1]) * u_sym[0] * self.problem.dyn_dict[
                                                     'u_max'] + v_curr_func(ca.vertcat(x_sym[1], x_sym[0]))) /
                                                 self.sim_settings['conv_m_to_deg'],
                                                 charge - self.problem.dyn_dict['energy'] *
                                                 (self.problem.dyn_dict['u_max'] * u_sym[0]) ** 3,
                                                 1)],
                                     ['x', 'u'], ['x_dot'])

        # create an integrator out of it
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

        # return F_next for future use
        return F_x_next

    def run(self, T_in_h=None, max_steps=500):
        """Main Loop of the simulator including replanning etc.
        The Loop runs with step-size dt for time T_in_h or until the goal is reached (or max_steps is reached)
        Returns success boolean.

        # TODO: for now there is one main loop, at some point we want to have parallel processes/workers
        running the high-level planner, tracking controller, simulator at their respective frequencies.
        This has 2 advantages: 1) the can run on actual frequencies and not limited by for loop, 2) speed up!
        """
        end_sim = False

        if self.sim_settings['plan_on_gt']:
            self.high_level_planner.update_forecast_file(
                self.problem.hindcast_file)
            # put something in so that the rest of the code runs
            current_forecast_dict_idx = 0
        else:
            current_forecast_dict_idx = self.problem.most_recent_forecast_idx
            self.high_level_planner.update_forecast_file(
                self.problem.forecasts_dict[current_forecast_dict_idx]['file'])

        # tracking variables
        next_planner_update = 0.
        next_tracking_controller_update = 0.

        while not end_sim:
            # Loop 1: update forecast files if new one is available
            if (not self.sim_settings['plan_on_gt']) and self.cur_state[3] >= \
                    self.problem.forecasts_dict[current_forecast_dict_idx + 1]['t_range'][0] \
                    + self.problem.forecast_delay_in_h * 3600.:
                self.high_level_planner.update_forecast_file(
                    self.problem.forecasts_dict[current_forecast_dict_idx + 1]['file'])
                current_forecast_dict_idx = current_forecast_dict_idx + 1
                # trigger high-level planner re-planning
                next_planner_update = self.cur_state[3]

            # Loop 2: replan with high-level planner
            if self.cur_state[3] >= next_planner_update:
                self.high_level_planner.plan(self.cur_state, trajectory=None)
                self.wypt_tracking_contr.set_waypoints(self.high_level_planner.get_waypoints())
                self.wypt_tracking_contr.replan(self.cur_state)
                # set new updating times
                next_planner_update = self.cur_state[3] \
                                      + self.control_settings['planner']['dt_replanning']
                next_tracking_controller_update = self.cur_state[3]\
                                                  + self.control_settings['waypoint_tracking']['dt_replanning']
                print("High-level planner & tracking controller replanned")

            # Loop 3: replanning of Waypoint tracking
            if self.cur_state[3] >= next_tracking_controller_update:
                self.wypt_tracking_contr.replan(self.cur_state)
                next_tracking_controller_update = self.cur_state[3] \
                                                  + self.control_settings['waypoint_tracking']['dt_replanning']
                print("Tracking controller replanned")

            # simulator step
            self.run_step()
            print("Sim step")

            end_sim = self.check_termination(T_in_h, max_steps)
        return self.reached_goal()

    def run_step(self):
        """Run the simulator for one dt step"""

        u = self.thrust_check(self.wypt_tracking_contr.get_next_action(self.cur_state))

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
                       self.problem.dyn_dict['energy']*(self.problem.dyn_dict['u_max'] * u_planner[0]) ** 3

        next_charge = self.cur_state[2] + delta_charge*self.sim_settings['dt']

        # if smaller than 0.: change the thrust accordingly
        if next_charge < 0.:
            energy_available = self.cur_state[2]
            u_planner[0] = ((charge - energy_available/self.sim_settings['dt'])/\
                    (self.problem.dyn_dict['energy']*self.problem.dyn_dict['u_max']**3))**(1./3)
            return u_planner
        else:
            return u_planner

    def battery_check(self, cur_state):
        """Prevents battery level to go above 1."""
        if cur_state[2] > 1.:
            cur_state[2] = 1.
        return cur_state

    def check_termination(self, T_in_h, max_steps):
        """Helper function returning boolean.
        If False: simulation is continued, if True it ends."""
        t_sim_run = (self.cur_state[3] - self.problem.x_0[3])

        if self.reached_goal():
            print("Sim terminate because goal reached")
            return True
        if (T_in_h is not None) and (t_sim_run/ 3600.) > T_in_h:
            print("Sim terminate because T_in_h is over")
            return True
        if t_sim_run > max_steps * self.sim_settings['dt']:
            print("Sim terminate because max_steps reached")
            return True
        if self.cur_state[3] > self.grids_dict['t_grid'][-1]:
            print("Sim terminate because hindcast fieldset time is over")
            return True
        # check if we're going outside of the current sub-setted gt hindcasts
        # TODO: can build in a check if we can just re-do the sub-setting and keep going
        if (not self.grids_dict['x_grid'][0] <= self.cur_state[0] <= self.grids_dict['x_grid'][-1]) or \
            (not self.grids_dict['y_grid'][0] <= self.cur_state[1] <= self.grids_dict['y_grid'][-1]):
            print("Sim terminate because state went out of sub-setted gt hindcast file.")
            return True
        # TODO: implement a way of checking for stranding!
        # if self.cur_state in some region where there is land:
        #     print("Sim terminated because platform stranded on land")
        #     return True
        return False

    def reached_goal(self):
        """Returns whether we have reached the target goal """

        lon, lat = self.cur_state[0][0], self.cur_state[1][0]
        lon_target, lat_target = self.problem.x_T[0], self.problem.x_T[1]
        return abs(lon - lon_target) < self.sim_settings['slack_around_goal'] and abs(lat - lat_target) < \
               self.sim_settings['slack_around_goal']

    def plot_trajectory(self, plotting_type='2D', time_for_currents=None, html_render=None, vid_file_name=None):
        """ Captures the whole trajectory - energy, position, etc over time"""
        # process the time that is put in the for the 2D static plots
        if time_for_currents is None:
            time_for_currents = self.problem.x_0[3]
        else:
            time_for_currents = time_for_currents.timestamp()

        if plotting_type == '2D':
            plotting_utils.plot_2D_traj(self.trajectory[:2, :], title="Simulator Trajectory")
            return

        elif plotting_type == '2D_w_currents':
            print("Plotting 2D trajectory with the true currents at time_for_currents/t_0")
            plotting_utils.plot_2D_traj_over_currents(self.trajectory[:2, :],
                                                      time=time_for_currents,
                                                      file=self.problem.hindcast_file)
            return

        elif plotting_type == '2D_w_currents_w_controls':
            print("Plotting 2D trajectory with the true currents at time_for_currents/t_0")
            plotting_utils.plot_2D_traj_over_currents(self.trajectory[:2, :],
                                                      time=time_for_currents,
                                                      file=self.problem.hindcast_file,
                                                      ctrl_seq=self.control_traj,
                                                      u_max=self.problem.dyn_dict['u_max'])
            return

        elif plotting_type == 'ctrl':
            plotting_utils.plot_opt_ctrl(self.trajectory[3, :-1], self.control_traj,
                                         title="Simulator Control Trajectory")

        elif plotting_type == 'battery':
            fig, ax = plt.subplots(1, 1)
            # some stuff for flexible date axis
            locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            # plot
            dates = [datetime.datetime.utcfromtimestamp(posix) for posix in self.trajectory[3, :]]
            ax.plot(dates, self.trajectory[2, :])
            # set axis and stuff
            ax.set_title('Battery charge over time')
            ax.set_ylim(0., 1.1)
            plt.xlabel('time in h')
            plt.ylabel('Battery Charging level [0,1]')
            plt.show()
            return

        elif plotting_type == 'video':
            plotting_utils.plot_2D_traj_animation(
                traj_full=self.trajectory,
                control_traj=self.control_traj,
                file=self.problem.hindcast_file,
                u_max=self.problem.dyn_dict['u_max'],
                html_render=html_render, filename=vid_file_name)
