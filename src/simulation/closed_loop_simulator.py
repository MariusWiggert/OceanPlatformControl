import yaml
import parcels as p
import glob, imageio, os
from src.utils import simulation_utils, hycom_utils
import casadi as ca
import numpy as np
import src.planners as planners
import src.tracking_controllers as tracking_controllers
import matplotlib.pyplot as plt


class ClosedLoopSimulator:
    """
    This class takes in a simulator and control YAML file for configuration as well as a problem.
    It has access to the forecasted currents for each day and hindcasts over the time period.
    It simulates the closed-loop performance of the running-system.

    sim_config:                     Simulator YAML configuration file
    control_config:                 Controller YAML configuration file
    problem:                        Problem for the planner (x_0, x_T, t_0, radius, noise)
    project_dir (optional):         Directory of the project for access to config YAMLs and data
    """

    def __init__(self, sim_config, control_config, problem, project_dir=None):
        # get project directory
        if project_dir is None:
            self.project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        else:
            self.project_dir = project_dir
        # load simulator sim_settings YAML
        with open(self.project_dir + '/config/' + sim_config) as f:
            sim_settings = yaml.load(f, Loader=yaml.FullLoader)
            sim_settings['project_dir'] = project_dir

        # load controller settings YAML
        with open(self.project_dir + '/config/' + control_config) as f:
            control_settings = yaml.load(f, Loader=yaml.FullLoader)

        # check if update frequencies make sense:
        if control_settings['waypoint_tracking']['dt_replanning'] < sim_settings['dt']:
            raise ValueError("Simulator dt needs to be smaller than tracking controller replanning dt")

        self.sim_settings = sim_settings
        self.control_settings = control_settings
        self.problem = problem

        # initialize the GT dynamics (currently HYCOM Hindcasts, later potentially adding noise)
        # TODO: program a smart loader with input the problem and it checks if the data exists and loads it
        self.F_x_next = self.initialize_dynamics(self.problem.hindcast_file)
        # implement check if forecasts are available for those days of the problem
        # Note: for now we assume the predictions are ordered in time one daily

        # create instances for the high-level planner and a pointer to the waypoint tracking function
        # Step 1: initialize high-level planner
        planner_class = getattr(planners, self.control_settings['planner']['name'])
        self.high_level_planner = planner_class(self.problem,
                                                gen_settings=self.control_settings['planner']['gen_settings'],
                                                specific_settings=self.control_settings['planner']['specific_settings'])
        # Step 2: initialize the tracking controller
        if self.control_settings['waypoint_tracking']['name'] != 'None':
            tracking_class = getattr(tracking_controllers, self.control_settings['waypoint_tracking']['name'])
            self.wypt_tracking_contr = tracking_class()
        else:  # do open-loop control using the get_action_function from the high-level planner
            self.wypt_tracking_contr = self.high_level_planner
            print("Running Open-Loop Control without a tracking controller.")

        # initiate some tracking variables
        self.cur_state = np.array(problem.x_0).reshape(4, 1)  # lon, lat, battery level, time
        self.time_origin = problem.hindcast_fieldset.time_origin
        self.max_time_hindcast = max(problem.hindcast_fieldset.gridset.grids[0].time[:])
        self.trajectory = self.cur_state
        self.control_traj = []

    def initialize_dynamics(self, hindcast_file):
        """Initialize symbolic dynamics function for simulation
        Input: fieldset       a parcels fieldset

        Output:
            u_curr_func       interpolation function for longitude currents
            v_curr_func       interpolation function for latitude currents
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
            simulation_utils.get_current_data_subset(hindcast_file,
                                                     self.problem.x_0, self.problem.x_T,
                                                     self.sim_settings['deg_around_x0_xT_box'],
                                                     self.problem.fixed_time_index)

        # Step 2: get the current interpolation functions
        u_curr_func, v_curr_func = simulation_utils.get_interpolation_func(
            self.grids_dict, u_data, v_data, self.sim_settings['int_pol_type'], self.problem.fixed_time_index)

        # Step 3: create the x_dot dynamics function
        if self.problem.fixed_time_index is None:  # time varying current
            x_dot_func = ca.Function('f_x_dot', [x_sym, u_sym],
                                     [ca.vertcat((ca.cos(u_sym[1]) * u_sym[0] * self.problem.dyn_dict[
                                         'u_max'] + u_curr_func(ca.vertcat(x_sym[3], x_sym[1], x_sym[0]))) /
                                                 self.sim_settings['conv_m_to_deg'],
                                                 (ca.sin(u_sym[1]) * u_sym[0] * self.problem.dyn_dict[
                                                     'u_max'] + v_curr_func(
                                                     ca.vertcat(x_sym[3], x_sym[1], x_sym[0]))) / self.sim_settings[
                                                     'conv_m_to_deg'],
                                                 self.problem.dyn_dict['charge'] - self.problem.dyn_dict['energy'] *
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
                                                 self.problem.dyn_dict['charge'] - self.problem.dyn_dict['energy'] *
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

        # tracking variables
        # TODO: this will need to be a lookup list of when forecasts were available
        next_forecast_update_time = 0.
        next_tracking_controller_update = 0.

        while not end_sim:
            # TODO: flexible updating based on a dict
            # update forecast files daily and replan with planner and hand-over to tracking controller
            if self.cur_state[3] >= next_forecast_update_time:
                # self.high_level_planner.plan(self.cur_state, new_forecast_file=self.problem.forecast_list[0],
                #                              trajectory=None)
                self.high_level_planner.plan(self.cur_state, new_forecast_file=self.problem.hindcast_file,
                                             trajectory=None)
                self.wypt_tracking_contr.set_waypoints(self.high_level_planner.get_waypoints())
                self.wypt_tracking_contr.replan(self.cur_state)
                # next_forecast_update_time = self.cur_state[3] + 3600.*24
                next_forecast_update_time = self.cur_state[3] + 3600.*24*100
                next_tracking_controller_update = self.cur_state[3]\
                                                  + self.control_settings['waypoint_tracking']['dt_replanning']
                print("High-level planner & tracking controller replanned")

            # if self.cur_state[3] >= next_tracking_controller_update:
            #     self.wypt_tracking_contr.replan(self.cur_state)
            #     print("Tracking controller replanned")

            # simulator step
            self.run_step()
            print("Sim step")

            end_sim = self.check_termination(T_in_h, max_steps)
        return self.reached_goal()

    def run_step(self):
        """Run the simulator for one dt step"""

        u = self.thrust_check(self.wypt_tracking_contr.get_next_action(self.cur_state))

        # update simulator states
        self.control_traj.append(u)
        self.cur_state = self.battery_check(np.array(self.F_x_next(self.cur_state, u)).astype('float32'))
        # add new state to trajectory
        self.trajectory = np.hstack((self.trajectory, self.cur_state))

    def thrust_check(self, u_planner):
        """If the thrust would use more energy than available adjust accordingly."""

        delta_charge = self.problem.dyn_dict['charge'] - \
                       self.problem.dyn_dict['energy']*(self.problem.dyn_dict['u_max'] * u_planner[0]) ** 3

        next_charge = self.cur_state[2] + delta_charge*self.sim_settings['dt']

        # if smaller than 0.: change the thrust accordingly
        if next_charge < 0.:
            energy_available = self.cur_state[2]
            u_planner[0] = ((self.problem.dyn_dict['charge'] - energy_available/self.sim_settings['dt'])/\
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
        if T_in_h is not None and (self.cur_state[3] / 3600.) > T_in_h:
            return True
        if self.reached_goal():
            return True
        if self.cur_state[3] > max_steps * self.sim_settings['dt']:
            return True
        if self.cur_state[3] > self.max_time_hindcast:
            return True
        return False

    def reached_goal(self):
        """Returns whether we have reached the target goal """

        lon, lat = self.cur_state[0][0], self.cur_state[1][0]
        lon_target, lat_target = self.problem.x_T[0], self.problem.x_T[1]
        return abs(lon - lon_target) < self.sim_settings['slack_around_goal'] and abs(lat - lat_target) < \
               self.sim_settings['slack_around_goal']

    def plot_trajectory(self, name, plotting_type='2D'):
        """ Captures the whole trajectory - energy, position, etc over time
        Accesses the trajectory and fieldset from the problem.
        """

        if plotting_type == '2D':
            plt.figure(1)
            plt.plot(self.trajectory[0, :], self.trajectory[1, :], '--')
            plt.plot(self.trajectory[0, 0], self.trajectory[1, 0], '--', marker='x', color='red')
            plt.plot(self.trajectory[0, -1], self.trajectory[1, -1], '--', marker='x', color='green')
            plt.title('Simulated Trajectory of Platform')
            plt.xlabel('lon in deg')
            plt.ylabel('lat in deg')
            plt.show()
            return

        elif plotting_type == 'battery':
            plt.figure(1)
            plt.plot(self.trajectory[3, :]/3600., self.trajectory[2, :], '-')
            plt.title('Battery charge over time')
            plt.ylim(0.,1.1)
            plt.xlabel('time in h')
            plt.ylabel('Battery Charging level [0,1]')
            plt.show()
            return

        elif plotting_type == 'gif':
            # Step 1: create the images
            for i in range(self.trajectory.shape[1]):
                # under the assumption that x is a Position
                pset = p.ParticleSet.from_list(
                        fieldset=self.problem.hindcast_fieldset,  # the fields on which the particles are advected
                        pclass=p.ScipyParticle, # the type of particles (JITParticle or ScipyParticle)
                        lon=[self.trajectory[0,i]],  # a vector of release longitudes
                        lat=[self.trajectory[1,i]],  # a vector of release latitudes
                    )

                if self.problem.fixed_time_index is None:
                    pset.show(savefile=self.project_dir + '/viz/pics_2_gif/particles' + str(i).zfill(2),
                              field='vector', land=True,
                              vmax=1.0, show_time=self.trajectory[3, i])
                else:
                    pset.show(savefile=self.project_dir + '/viz/pics_2_gif/particles' + str(i).zfill(2),
                              field='vector', land=True,
                              vmax=1.0, show_time=self.problem.fieldset.gridset.grids[0].time[self.problem.fixed_time_index])

            # Step 2: compile to gif
            file_list = glob.glob(self.project_dir + "/viz/pics_2_gif/*")
            file_list.sort()

            gif_file = self.project_dir + '/viz/gifs/' + name + '.gif'
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in file_list:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    os.remove(filename)
            print("saved gif as " + name)
            return
