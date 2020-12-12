import parcels as p
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, Field, Variable
import glob, imageio, os
from src.utils import kernels, particles, optimal_control_utils
import casadi as ca
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


class Position:
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat


class ProblemSet:
    def __init__(self, fieldset):
        self.fieldset = fieldset

    def create_problem(self):
        """ TODO: Randomly generate Problems with valid x_0 and x_T """


class Problem:
    def __init__(self, fieldset, x_0, x_T, u_max):
        self.fieldset = fieldset                        
        self.x_0 = x_0                                  # Start Position
        self.x_T = x_T                                  # Final Position
        self.u_max = u_max                              # maximal actuation of the platform in m/s

    def viz(self):
        pset = p.ParticleSet.from_list(fieldset=self.fieldset,  # the fields on which the particles are advected
                                       pclass=p.ScipyParticle,
                                       # the type of particles (JITParticle or ScipyParticle)
                                       lon=[self.x_0[0], self.x_T[0]],  # a vector of release longitudes
                                       lat=[self.x_0[1], self.x_T[1]],  # a vector of release latitudes
                                       )
        pset.show(field='vector')


class Planner:
    """ All Planners should inherit this class """

    def get_next_action(self, state, rel_time):
        """ TODO: returns (thrust, header) for the next timestep """
        raise NotImplementedError()

    def transform_u_dir_to_u(self, u_dir):
        thrust = np.sqrt(u_dir[0]**2 + u_dir[1]**2)
        heading = np.arctan(u_dir[1]/u_dir[0])
        return np.array([thrust, heading])


class TrajectoryTrackingController:
    pass 


class Simulator:
    """
    Functions as our "Parcels". Users will:
    1. Create Simulator object with planner, problem, 
    2. Execute simulator.run()
    3. Fetch trajectory plot with plot_trajectory
    4. Evaluate planner solving given problem under some metric
    """
    def __init__(self, planner, problem, settings=None):
        # default settings
        if settings is None:
            settings = {'project_dir': 'text', 'dt': 1., 'conv_m_to_deg': 111120.,
                        'int_pol_type': 'bspline', 'sim_integration': 'rk'}
        self.planner = planner
        self.problem = problem
        self.cur_state = problem.x_0
        self.time_origin = problem.fieldset.time_origin
        self.rel_time = timedelta(seconds=0.)
        self.trajectory = np.array(problem.x_0).reshape(-1, 1)
        # self.control_traj = []            # """TODO: implement this mainly for debugging & viz"""
        self.settings = settings

        # initialize dynamics
        self.u_curr_func, self.v_curr_func, self.F_x_next = self.initialize_dynamics()

    def run(self, T):
        """ runs the simulator for time T"""
        # run over T with dt stepsize
        N = int(T // self.settings['dt']) + 1
        print('N',N)
        for _ in range(N):
            # get next action
            u = self.planner.get_next_action(self.cur_state, self.rel_time.total_seconds())
            # update simulator states
            self.cur_state = np.array(self.F_x_next(self.cur_state, u))
            self.rel_time += timedelta(seconds=self.settings['dt'])
            # add new state to trajectory
            self.trajectory = np.hstack((self.trajectory, self.cur_state))

    def run_step(self):
        # run one dt step
        u = self.planner.get_next_action(self.cur_state, self.rel_time.total_seconds())
        # update simulator states
        self.cur_state = np.array(self.F_x_next(self.cur_state, u))
        self.rel_time += timedelta(seconds=self.settings['dt'])
        # add new state to trajectory
        self.trajectory = np.hstack((self.trajectory, self.cur_state))

    def initialize_dynamics(self):
        """ Initialize symbolic dynamics function for simulation"""
        """ TODO: 
        - add 3rd & 4th state (battery, time) 
        - make input heading & trust!
        """
        x_sym_1 = ca.MX.sym('x1')
        x_sym_2 = ca.MX.sym('x2')
        x_sym = ca.vertcat(x_sym_1, x_sym_2)

        u_sim_1 = ca.MX.sym('u_1')
        u_sim_2 = ca.MX.sym('u_2')
        u_sym = ca.vertcat(u_sim_1, u_sim_2)

        # get the current interpolation functions
        u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(
            self.problem.fieldset, self.settings['conv_m_to_deg'], type=self.settings['int_pol_type'])

        # create the x_dot dynamics function
        x_dot_func = ca.Function('f_x_dot', [x_sym, u_sym],
                                   [ca.vertcat(u_sym[0] / self.settings['conv_m_to_deg'] + u_curr_func(ca.vertcat(x_sym[0], x_sym[1])),
                                               u_sym[1] / self.settings['conv_m_to_deg'] + v_curr_func(ca.vertcat(x_sym[0], x_sym[1])))],
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
                                       [ca.vertcat(x_sym[0] + self.settings['dt'] * x_dot_func(x_sym, u_sym)[0],
                                                   x_sym[1] + self.settings['dt'] * x_dot_func(x_sym, u_sym)[1])],
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
            plt.title('Simulated Trajectory')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

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

                pset.show(savefile=self.settings['project_dir'] + '/viz/pics_2_gif/particles' + str(i).zfill(2), field='vector', land=True,
                    vmax=1.0, show_time=0.)

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
    
    def evaluate(self, planner, problem):
        """ TODO: Evaluate the planner on the given problem by some metrics """