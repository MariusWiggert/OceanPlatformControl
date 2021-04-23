import yaml
import parcels as p
import glob, imageio, os
from src.utils import simulation_utils
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class Simulator:
    """
    Functions as our "Parcels". Users will:
    1. Create Simulator object with planner, problem,
    2. Execute simulator.run()
    3. Fetch trajectory plot with plot_trajectory
    4. Evaluate planner solving given problem under some metric
    """
    def __init__(self, planner, problem, project_dir, sim_config):
        # load simulator settings
        with open(project_dir + '/config/' + sim_config) as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
            settings['project_dir'] = project_dir
        self.settings = settings

        self.planner = planner
        self.problem = problem
        self.cur_state = np.array(problem.x_0).reshape(4, 1)   # lon, lat, battery level, time
        self.time_origin = problem.fieldset.time_origin
        self.trajectory = self.cur_state
        self.control_traj = []

        # initialize dynamics
        self.u_curr_func, self.v_curr_func, self.F_x_next = self.initialize_dynamics()

    def run(self, T=None, max_steps=500):
        """ If T is None, runs the simulator for time T in seconds. Runs the planner until the goal is reached or the
        time exceeds timeout. Returns success boolean """
        # run over T with dt stepsize
        if T:
            N = int(T // self.settings['dt']) + 1
            for _ in range(N):
                self.run_step()
        else:
            step = 0
            while not self.reached_goal() and step < max_steps:
                self.run_step()
                step += 1
        return self.reached_goal()

    def reached_goal(self, slack=0.1):
        """Returns whether we have reached the target goal """

        lon, lat = self.cur_state[0][0], self.cur_state[1][0]
        lon_target, lat_target = self.problem.x_T[0], self.problem.x_T[1]
        return abs(lon - lon_target) < slack and abs(lat - lat_target) < slack

    def run_step(self):
        """Run the simulator for one dt step"""

        u = self.thrust_check(self.planner.get_next_action(self.cur_state))
        # update simulator states
        self.control_traj.append(u)
        self.cur_state = self.battery_check(np.array(self.F_x_next(self.cur_state, u)).astype('float32'))
        # add new state to trajectory
        self.trajectory = np.hstack((self.trajectory, self.cur_state))

    def thrust_check(self, u_planner):
        """If the thrust would use more energy than available adjust accordingly."""

        delta_charge = self.problem.dyn_dict['charge'] - \
                       self.problem.dyn_dict['energy']*(self.problem.dyn_dict['u_max'] * u_planner[0]) ** 3

        next_charge = self.cur_state[2] + delta_charge*self.settings['dt']

        # if smaller than 0.: change the thrust accordingly
        if next_charge < 0.:
            energy_available = self.cur_state[2]
            u_planner[0] = ((self.problem.dyn_dict['charge'] - energy_available/self.settings['dt'])/\
                    (self.problem.dyn_dict['energy']*self.problem.dyn_dict['u_max']**3))**(1./3)
            return u_planner
        else:
            return u_planner

    def battery_check(self, cur_state):
        """Prevents battery level to go above 1."""
        if cur_state[2] > 1.:
            cur_state[2] = 1.
        return cur_state

    def initialize_dynamics(self):
        """Initialize symbolic dynamics function for simulation """

        # Step 1: define variables
        x_sym_1 = ca.MX.sym('x1')   # lon
        x_sym_2 = ca.MX.sym('x2')   # lat
        x_sym_3 = ca.MX.sym('x3')   # battery
        x_sym_4 = ca.MX.sym('t')    # time
        x_sym = ca.vertcat(x_sym_1, x_sym_2, x_sym_3, x_sym_4)

        u_sim_1 = ca.MX.sym('u_1')  # thrust magnitude in [0,1]
        u_sim_2 = ca.MX.sym('u_2')  # header in radians
        u_sym = ca.vertcat(u_sim_1, u_sim_2)

        # Step 2: get the current interpolation functions
        u_curr_func, v_curr_func = simulation_utils.get_interpolation_func(
            self.problem.fieldset, type=self.settings['int_pol_type'], fixed_time_index=self.problem.fixed_time_index)

        # Step 3: create the x_dot dynamics function
        if self.problem.fixed_time_index is None:    # time varying current
            x_dot_func = ca.Function('f_x_dot', [x_sym, u_sym],
                                     [ca.vertcat((ca.cos(u_sym[1])*u_sym[0]*self.problem.dyn_dict['u_max'] + u_curr_func(ca.vertcat(x_sym[3], x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                 (ca.sin(u_sym[1])*u_sym[0]*self.problem.dyn_dict['u_max'] + v_curr_func(ca.vertcat(x_sym[3], x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                 self.problem.dyn_dict['charge'] - self.problem.dyn_dict['energy'] *
                                                 (self.problem.dyn_dict['u_max'] * u_sym[0]) ** 3,
                                                 1)],
                                     ['x', 'u'], ['x_dot'])
        else:   # fixed current
            x_dot_func = ca.Function('f_x_dot', [x_sym, u_sym],
                                     [ca.vertcat((ca.cos(u_sym[1])*u_sym[0]*self.problem.dyn_dict['u_max'] + u_curr_func(ca.vertcat(x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                 (ca.sin(u_sym[1])*u_sym[0]*self.problem.dyn_dict['u_max'] + v_curr_func(ca.vertcat(x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                 self.problem.dyn_dict['charge'] - self.problem.dyn_dict['energy'] *
                                                 (self.problem.dyn_dict['u_max'] * u_sym[0]) ** 3,
                                                 1)],
                                     ['x', 'u'], ['x_dot'])

        # create an integrator out of it
        if self.settings['sim_integration'] == 'rk':
            dae = {'x': x_sym, 'p': u_sym, 'ode': x_dot_func(x_sym, u_sym)}
            integ = ca.integrator('F_int', 'rk', dae, {'tf': self.settings['dt']})
            # Simplify API to (x,u)->(x_next)
            F_x_next = ca.Function('F_x_next', [x_sym, u_sym],
                                   [integ(x0=x_sym, p=u_sym)['xf']], ['x', 'u'], ['x_next'])

        elif self.settings['sim_integration'] == 'ef':
            F_x_next = ca.Function('F_x_next', [x_sym, u_sym],
                                       [x_sym + self.settings['dt'] * x_dot_func(x_sym, u_sym)],
                                       ['x', 'u'], ['x_next'])
        else:
            raise ValueError('sim_integration: only RK4 (rk) and forward euler (ef) implemented')

        # safe for future use
        return u_curr_func, v_curr_func, F_x_next

    def plot_trajectory(self, name, plotting_type='2D'):
        """ Captures the whole trajectory - energy, position, etc over time
        Accesses the trajectory and fieldset from the problem.
        """

        if plotting_type == '2D':
            plt.figure(1)
            plt.plot(self.trajectory[0, :], self.trajectory[1, :], '--')
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
                        fieldset=self.problem.fieldset,  # the fields on which the particles are advected
                        pclass=p.ScipyParticle, # the type of particles (JITParticle or ScipyParticle)
                        lon=[self.trajectory[0,i]],  # a vector of release longitudes
                        lat=[self.trajectory[1,i]],  # a vector of release latitudes
                    )

                if self.problem.fixed_time_index is None:
                    pset.show(savefile=self.settings['project_dir'] + '/viz/pics_2_gif/particles' + str(i).zfill(2),
                              field='vector', land=True,
                              vmax=1.0, show_time=self.trajectory[3, i])
                else:
                    pset.show(savefile=self.settings['project_dir'] + '/viz/pics_2_gif/particles' + str(i).zfill(2),
                              field='vector', land=True,
                              vmax=1.0, show_time=self.problem.fieldset.gridset.grids[0].time[self.problem.fixed_time_index])

            # Step 2: compile to gif
            file_list = glob.glob(self.settings['project_dir'] + "/viz/pics_2_gif/*")
            file_list.sort()

            gif_file = self.settings['project_dir'] + '/viz/gifs/' + name + '.gif'
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in file_list:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    os.remove(filename)
            print("saved gif as " + name)
            return

