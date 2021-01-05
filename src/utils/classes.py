import math
import pickle
import random
import parcels as p
import glob
import imageio
import os

from src.utils import simulation_utils, hycom_utils
import casadi as ca
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt


class ProblemSet:
    def __init__(self, fieldset, filename=None, num_problems=100):
        self.fieldset = fieldset
        if filename is None:
            self.problems = [self.create_problem() for _ in range(num_problems)]
        else:
            self.problems = self.load_problems(filename)

    def create_problem(self, u_max=.2):
        """ Randomly generates a Problem with valid x_0 and x_T """
        x_0, x_T = None, None
        while not self.valid_start_and_end(x_0, x_T):
            x_0, x_T = self.random_point(), self.random_point()
        return Problem(self.fieldset, x_0, x_T, u_max)

    def random_point(self):
        """ Returns a random point anywhere in the grid """
        lon = random.choice(self.fieldset.U.grid.lon)
        lat = random.choice(self.fieldset.U.grid.lat)
        return [lon, lat]

    def valid_start_and_end(self, x_0, x_T):
        """ Returns whether the given start (x_0) and end (x_T) is valid """
        if x_0 is None or x_T is None:
            return False
        return self.within_range(x_0, x_T) and self.in_ocean(x_0) and self.in_ocean(x_T)

    def within_range(self, x_0, x_T, min_sep=0.2, max_sep=1.5):
        """ Returns whether x_0 and x_T are not too close (min_sep) and not too far (max_sep) """
        lon, lat, lon_target, lat_target = x_0[0], x_0[1], x_T[0], x_T[1]
        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)
        return min_sep < mag < max_sep

    def in_ocean(self, point, offset=0.1):
        """ Returns whether a point is in the ocean. Determines this by checking if the velocity is nonzero for this
        and ALL points that are "offset" distance about the point in the 8 directions. """

        lon, lat = point[0], point[1]
        offsets = [(0, 0), (0, offset), (offset, 0), (offset, offset), (0, -offset),
                   (-offset, 0), (-offset, -offset), (offset, -offset), (-offset, offset)]
        for lon_offset, lat_offset in offsets:
            if self.zero_velocity(lon + lon_offset, lat + lat_offset):
                return False
        return True

    def out_of_bounds(self, coordinate, grid):
        """ Returns whether the given coordinate (either lat or lon) is out of bounds for its respective grid """
        return coordinate < min(grid) or coordinate > max(grid)

    def zero_velocity(self, lon, lat):
        """ Returns whether the (lon, lat) pair is zero velocity, i.e. on land """
        if self.out_of_bounds(lat, self.fieldset.U.grid.lat) or self.out_of_bounds(lon, self.fieldset.U.grid.lon):
            return False
        x = self.fieldset.U.eval(0., 0., lat, lon)
        y = self.fieldset.V.eval(0., 0., lat, lon)
        return x == 0. and y == 0.

    def load_problems(self, filename):
        """ Deserializes the list of problems from the filename """
        return pickle.load(open(filename))

    def save_problems(self, filename):
        """ Serializes the list of problems to the filename """
        pickle.dump(self.problems, open(filename, 'w'))

class Problem:
    def __init__(self, fieldset, x_0, x_T, u_max, charging=1., bat_cap=100., fixed_time_index=None):
        self.fieldset = fieldset
        # check if we do a fixed or variable current problem
        if fieldset.U.grid.time.shape[0] == 1:
            self.fixed_time_index = 0
        else:
            self.fixed_time_index = fixed_time_index
        if len(x_0) == 2:   # add 100% charge and time
            x_0 = x_0 + [bat_cap, 0.]
        elif len(x_0) == 3:  # add time only
            x_0.append(0.)
        self.x_0 = x_0                                  # Start State
        self.x_T = x_T                                  # Final Position (only lon, lat)
        self.u_max = u_max                              # maximal actuation of the platform in m/s
        self.charging = charging                        # charging of the battery capacity per second
        self.battery_capacity = bat_cap                 # battery capacity

    def viz(self):
        """TODO: Implement show_time"""
        pset = p.ParticleSet.from_list(fieldset=self.fieldset,  # the fields on which the particles are advected
                                       pclass=p.ScipyParticle,
                                       # the type of particles (JITParticle or ScipyParticle)
                                       lon=[self.x_0[0], self.x_T[0]],  # a vector of release longitudes
                                       lat=[self.x_0[1], self.x_T[1]],  # a vector of release latitudes
                                       )
        pset.show(field='vector')


class Planner:
    """ All Planners should inherit this class """

    def __init__(self, problem, settings, t_init, n, mode='open-loop', fixed_time_index=None):

        if settings is None:
            settings = {'conv_m_to_deg': 111120., 'int_pol_type': 'bspline', 'dyn_constraints': 'ef'}
        # problem containing the vector field, x_0, and x_T
        self.problem = problem

        # time to run the fixed time optimal control problem
        self.T_init = t_init
        # number of decision variables in the trajectory
        self.N = n
        self.settings = settings
        self.mode = mode

    def get_next_action(self, state):
        """ TODO: returns (thrust, header) for the next timestep """
        raise NotImplementedError()

    def transform_u_dir_to_u(self, u_dir):
        thrust = np.sqrt(u_dir[0]**2 + u_dir[1]**2)         # Calculating thrust from distance formula on input u
        heading = np.arctan2(u_dir[1], u_dir[0])              # Finds heading angle from input u
        return np.array([thrust, heading])


class TrajectoryTrackingController:
    """
    TODO: implement controller to go along planned trajectory for platforms.
    """
    def __init(self, trajectory, state): # Will need planned trajectory and current state
        pass

class Simulator:
    """
    Functions as our "Parcels". Users will:
    1. Create Simulator object with planner, problem
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
        self.cur_state = np.array(problem.x_0).reshape(4, 1)   # lon, lat, battery level, time
        self.time_origin = problem.fieldset.time_origin
        self.trajectory = self.cur_state
        # self.control_traj = []            # """TODO: implement this mainly for debugging & viz"""
        self.settings = settings

        # initialize dynamics
        self.u_curr_func, self.v_curr_func, self.F_x_next = self.initialize_dynamics()

    def reset(self, planner, problem):
        """ Resets the state of the simulator """
        self.planner, self.problem = planner, problem
        self.cur_state = np.array(problem.x_0).reshape(4, 1)   # lon, lat, battery level, time
        self.time_origin = problem.fieldset.time_origin
        self.trajectory = self.cur_state

    def reached_goal(self, slack=0.1):
        """ Returns whether we have reached the target goal """
        lon, lat = self.cur_state[0][0], self.cur_state[1][0]
        lon_target, lat_target = self.problem.x_T[0], self.problem.x_T[1]
        return abs(lon - lon_target) < slack and abs(lat - lat_target) < slack

    def run_until_success(self, max_steps=100000):
        """ Runs the planner until the goal is reached or the time exceeds timeout. Returns (success, time) """
        step = 0
        while not self.reached_goal() and step < max_steps:
            self.run_step()
            step += 1
        return self.reached_goal(), step * self.settings['dt']

    def run(self, T):
        """ Runs the simulator for time T"""
        # run over T with dt stepsize
        N = int(T // self.settings['dt']) + 1
        for _ in range(N):
            self.run_step()

    def run_step(self):
        """ Run one dt step """
        u = self.planner.get_next_action(self.cur_state)
        # update simulator states
        self.cur_state = np.array(self.F_x_next(self.cur_state, u)).astype('float32')
        # add new state to trajectory
        self.trajectory = np.hstack((self.trajectory, self.cur_state))

    def initialize_dynamics(self):
        """ Initialize symbolic dynamics function for simulation"""
        """ TODO:
        - add 3rd & 4th state (battery, time)
        - make input heading & trust!
        """
        # Step 1: define variables
        x_sym_1 = ca.MX.sym('x1')   # lon
        x_sym_2 = ca.MX.sym('x2')   # lat
        x_sym_3 = ca.MX.sym('x3')   # battery
        x_sym_4 = ca.MX.sym('t')    # time
        x_sym = ca.vertcat(x_sym_1, x_sym_2, x_sym_3, x_sym_4)

        u_sim_1 = ca.MX.sym('u_1')  # thrust
        u_sim_2 = ca.MX.sym('u_2')  # header
        u_sym = ca.vertcat(u_sim_1, u_sim_2)

        # Step 2: get the current interpolation functions
        u_curr_func, v_curr_func = simulation_utils.get_interpolation_func(
            self.problem.fieldset, type=self.settings['int_pol_type'], fixed_time_index=self.problem.fixed_time_index)

        # Step 3: create the x_dot dynamics function
        if self.problem.fixed_time_index is None:    # time varying current
            x_dot_func = ca.Function('f_x_dot', [x_sym, u_sym],
                                       [ca.vertcat((ca.cos(u_sym[1])*u_sym[0] + u_curr_func(ca.vertcat(x_sym[4], x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                   (ca.sin(u_sym[1])*u_sym[0] + v_curr_func(ca.vertcat(x_sym[4], x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                   self.problem.charging, # implement taking of energy - u_sym[0]**3
                                                   1)],
                                       ['x', 'u'], ['x_dot'])
        else:   # fixed current
            x_dot_func = ca.Function('f_x_dot', [x_sym, u_sym],
                                     [ca.vertcat((ca.cos(u_sym[1])*u_sym[0] + u_curr_func(ca.vertcat(x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                 (ca.sin(u_sym[1])*u_sym[0] + v_curr_func(ca.vertcat(x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                 self.problem.charging,
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


class EvaluatePlanners:

    def __init__(self, problem_set):
        self.problem_set = problem_set
        self.successes = {}         # map each planner to a percentage of successes as a decimal between 0 and 1
        self.times = {}             # map each planner to a list of times
        self.battery_levels = []    # map each planner to a list of lists of battery level

    def evaluate_planner(self, planner):
        problem_set = ProblemSet(fieldset=self.problem.fieldset, )
        total_successes, total_times, total_battery_levels = 0, [], []
        for problem in problem_set.problems:
            planner.problem = problem
            success, time, battery_levels = self.evaluate_on_problem(planner=planner, problem=problem)
            total_successes += success
            if success:
                total_times.append(time)
                total_battery_levels.append(battery_levels)
        self.successes[planner] = total_successes / len(problem_set.problems)
        self.times[planner] = total_times
        self.battery_levels[planner] = total_battery_levels

    def evaluate_on_problem(self, planner, problem, settings=None):
        """ Evaluate the planner on the given problem by some metrics.
        Currently returns (success, time, percent of time below battery level threshold))
        """

        # Step 1: create the simulator
        sim = Simulator(planner, problem, settings)

        # Step 2: run the planner on the problem
        success, steps = sim.run_until_success()
        if not success:
            return False, None, (None, None, None)

        # Step 3: extract the "time" variable of the final state, which is the last element of the last list
        time = sim.trajectory[-1][-1]

        # Step 4: extract a list of all the battery levels
        battery_levels = sim.trajectory[2]

        return success, time, battery_levels

    def extract_battery_level_data(self, battery_levels, bat_level_threshold=0.2):

        average = np.average(battery_levels)
        variance = np.var(battery_levels)

        # we will create a list of all the battery levels below the threshold to calculate percentage of time below
        below_thresh = list(filter(lambda bat_level: bat_level < bat_level_threshold, battery_levels))
        percent_below = len(below_thresh) / len(battery_levels)

        return average, variance, percent_below
