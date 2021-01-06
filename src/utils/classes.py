"""The underlying classes for the Ocean Platform.

This module contains the implementation of the following classes — Problem, ProblemSet, Planner,
TrajectoryTrackingController, Simulator, and EvaluatePlanners. Note that the Planner class provides a protocol for a
planner, e.g. StraightLinePlanner, to inherit.
"""

import math
import pickle
import random

import yaml
import parcels as p
import glob, imageio, os
from src.utils import simulation_utils, hycom_utils
import casadi as ca
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt


class Problem:
    """A path planning problem for a Planner to solve.

    Attributes:
        fieldset:
            The fieldset contains data about ocean currents and is more rigorously defined in the
            parcels documentation.
        x_0:
            The starting state, represented as (lon, lat, battery_level, time).
        x_T:
            The target state, represented as (lon, lat).
        project_dir:
            A string giving the path to the project directory.
        config_yaml:
            A YAML file for the Problem configurations.
        fixed_time_index:
            An index of a fixed time. Note that the fixed_time_index is only
            applicable when the time is fixed, i.e. the currents are not time varying.
    """
    def __init__(self, fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=None):
        self.fieldset = fieldset
        if fieldset.U.grid.time.shape[0] == 1:  # check if we do a fixed or variable current problem
            self.fixed_time_index = 0
        else:
            self.fixed_time_index = fixed_time_index
        if len(x_0) == 2:  # add 100% charge and time
            x_0 = x_0 + [1., 0.]
        elif len(x_0) == 3:  # add time only
            x_0.append(0.)
        self.x_0 = x_0
        self.x_T = x_T
        self.dyn_dict = self.derive_platform_dynamics(project_dir, config=config_yaml)

    def __repr__(self):
        """Returns the string representation of a Problem, to be used for debugging.

        Returns:
            A String
        """
        return "Problem(x_0: {0}, x_T: {1})".format(self.x_0, self.x_T)

    def viz(self):
        """Visualizes the problem in a plot.

        Returns:
            None
        """
        pset = p.ParticleSet.from_list(fieldset=self.fieldset,  # the fields on which the particles are advected
                                       pclass=p.ScipyParticle,  # the type of particles (JITParticle or ScipyParticle)
                                       lon=[self.x_0[0], self.x_T[0]],  # a vector of release longitudes
                                       lat=[self.x_0[1], self.x_T[1]],  # a vector of release latitudes
                                       )
        pset.show(field='vector')

    def derive_platform_dynamics(self, project_dir, config):
        """Derives battery capacity dynamics for the specific dt.

        Args:
            project_dir:
                See class docstring
            config:
                See config_yaml in class docstring

        Returns:
            A dictionary of settings for the Problem, i.e. {'charge': __, 'energy': __, 'u_max': __}
        """

        # load in YAML
        with open(project_dir + '/config/' + config) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            platform_specs = config_data['platform_config']

        # derive calculation
        cap_in_joule = platform_specs['battery_cap']*3600
        energy_coeff = (platform_specs['drag_factor'] * (1/platform_specs['motor_efficiency']))/cap_in_joule
        charge_factor = platform_specs['avg_solar_power'] / cap_in_joule
        platform_dict = {'charge': charge_factor, 'energy': energy_coeff, 'u_max': platform_specs['u_max']}

        return platform_dict


class ProblemSet:
    """Stores a list of Problems.

    If no filename is given, the list of problems is randomly created. Else, the list of problems is deserialized
    from the filename with pickle. This class is used in the EvaluatePlanner's evaluate method to provide a collection
    of Problems to test a given Planner.

    Attributes:
        fieldset:
            The fieldset contains data about ocean currents and is more rigorously defined in the
            parcels documentation.
        project_dir:
            A string giving the path to the project directory.
        problems:
            A list of Problems.
    """
    def __init__(self, fieldset, project_dir, filename=None, num_problems=100):
        self.fieldset = fieldset
        self.project_dir = project_dir
        if filename is None:
            # random.seed(num_problems)
            self.problems = [self.create_problem() for _ in range(num_problems)]
        else:
            self.problems = self.load_problems(filename)

    def create_problem(self):
        """Randomly generates a Problem with valid x_0 and x_T.

        Iteratively produces random problems until one that fulfills the criteria in valid_start_and_end is found.

        Returns:
            A Problem.
        """
        x_0, x_T = None, None
        while not self.valid_start_and_end(x_0, x_T):
            x_0, x_T = self.random_point(), self.random_point()
        return Problem(self.fieldset, x_0, x_T, self.project_dir)

    def random_point(self):
        """Returns a random point anywhere in the grid.

        Returns:
            A point, i.e. a pair of longitude and latitude coordinates: [lon, lat].
        """
        lon = random.choice(self.fieldset.U.grid.lon)
        lat = random.choice(self.fieldset.U.grid.lat)
        return [lon, lat]

    def valid_start_and_end(self, x_0, x_T):
        """Determines whether the given start (x_0) and target (x_T) are valid.

        For a start and end to be valid, they must be sufficiently far apart, and neither point can be in the ocean.

        Args:
            x_0:
                The starting point, a pair of longitude and latitude coordinates: [lon, lat].
            x_T:
                The target point, a pair of longitude and latitude coordinates: [lon, lat].
        Returns:
            A boolean.
        """
        if x_0 is None or x_T is None:
            return False
        return self.is_far_apart(x_0, x_T) and self.in_ocean(x_0) and self.in_ocean(x_T)

    def is_far_apart(self, x_0, x_T, sep=0.5):
        """Returns whether x_0 and x_T are sufficiently far apart

        Args:
            x_0:
                The starting point, a pair of longitude and latitude coordinates: [lon, lat].
            x_T:
                The target point, a pair of longitude and latitude coordinates: [lon, lat].
            sep:
                The minimum distance between the two points.
        Returns:
            A boolean.
        """
        lon, lat, lon_target, lat_target = x_0[0], x_0[1], x_T[0], x_T[1]
        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)  # mag is the distance between the two points.
        return mag > sep

    def in_ocean(self, point, offset=0.1):
        """ Returns whether the point is in the ocean.

        Determines this by checking if the velocity is nonzero for this and ALL points that are "offset" distance
        about the point in the 8 directions.

        Args:
            point:
                A pair of longitude and latitude coordinates: [lon, lat].
            offset: A float which determines how far about the point to look. Increasing the value of offset will
                prevent points on the coast from being chosen.

        Returns:
            A boolean.
        """
        lon, lat = point[0], point[1]
        offsets = [(0, 0), (0, offset), (offset, 0), (offset, offset), (0, -offset),
                   (-offset, 0), (-offset, -offset), (offset, -offset), (-offset, offset)]
        for lon_offset, lat_offset in offsets:
            if self.zero_velocity(lon + lon_offset, lat + lat_offset):
                return False
        return True

    def out_of_bounds(self, coordinate, grid):
        """Determines whether the given coordinate (either lat or lon) is out of bounds for its respective grid.

        Returns:
            A boolean.
        """
        return coordinate < min(grid) or coordinate > max(grid)

    def zero_velocity(self, lon, lat):
        """Determines whether the (lon, lat) pair is zero velocity, i.e. on land.

        Returns:
            A boolean.
        """
        if self.out_of_bounds(lat, self.fieldset.U.grid.lat) or self.out_of_bounds(lon, self.fieldset.U.grid.lon):
            return False
        x = self.fieldset.U.eval(0., 0., lat, lon)
        y = self.fieldset.V.eval(0., 0., lat, lon)
        return x == 0. and y == 0.

    def load_problems(self, filename):
        """Deserializes the list of problems from the filename.

        Args:
            filename:
                A filename represented as a String, e.g. 'file.txt', that need not already exist.
        Returns:
            A list of Problems.
        """
        with open(filename, 'rb') as reader:
            return pickle.load(reader)

    def save_problems(self, filename):
        """Serializes the list of problems to the filename.

        Returns:
            None
        """
        with open(filename, 'wb') as writer:
            pickle.dump(self.problems, writer)


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

    def __repr__(self):
        return "Planner(mode: {0}, problem: {1})".format(self.mode, self.problem)

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

    def run(self, T=None, max_steps=100000):
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
                                     [ca.vertcat((ca.cos(u_sym[1])*u_sym[0]*self.problem.dyn_dict['u_max'] + u_curr_func(ca.vertcat(x_sym[4], x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
                                                 (ca.sin(u_sym[1])*u_sym[0]*self.problem.dyn_dict['u_max'] + v_curr_func(ca.vertcat(x_sym[4], x_sym[1], x_sym[0])))/self.settings['conv_m_to_deg'],
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
            plt.title('Simulated Trajectory')
            plt.xlabel('x')
            plt.ylabel('y')
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
            return


class EvaluatePlanners:
    """Evaluates a planner on a set of problems

    Attributes:
        problem_set:
            A ProblemSet instance which will supply the problems.
        project_dir:
            A string giving the path to the project directory.
        total_successes:
            A dictionary that maps each planner to the proportion of successes as a decimal between 0 and 1. For this,
            and the following two dictionaries, planners are only added after the evaluate_planner method is called.
        all_times:
            A dictionary that maps each planner to a list of times, one time for each Problem solved.
        all_battery_levels:
            A dictionary that maps each planner to a list of lists of battery levels, one list for each Problem solved.
        failed_problems:
            A dictionary that maps each planner to a list of Problems it failed.
    """
    def __init__(self, problem_set, project_dir):
        self.problem_set = problem_set
        self.project_dir = project_dir
        self.total_successes = {}
        self.all_times = {}
        self.all_battery_levels = {}
        self.failed_problems = {}

    def evaluate_planner(self, planner):
        """ Evaluates the planner on all the problems in self.problem_set.

        Calls self.evaluate_on_problem(...) for each problem in the problem_set. After looping through all the problems,
        populates total_successes, all_times, all_battery_levels, and failed_problems with the pertinent data.

        Args:
            planner:
                An instance of a subclass that inherits Planner, e.g. StraightLinePlanner

        Returns:
            None
        """
        total_successes, total_times, total_battery_levels, failed_problems = 0, [], [], []
        for problem in self.problem_set.problems:
            success, time, battery_levels = self.evaluate_on_problem(planner=planner, problem=problem)
            total_successes += success
            if success:
                total_times.append(time)
                total_battery_levels.append(battery_levels)
            else:
                failed_problems.append(problem)
        self.total_successes[planner] = total_successes / len(self.problem_set.problems)
        self.all_times[planner] = total_times
        self.all_battery_levels[planner] = total_battery_levels
        self.failed_problems[planner] = failed_problems

    def evaluate_on_problem(self, planner, problem, sim_config='simulator.yaml'):
        """Evaluate the planner on the given problem by some metrics.

        Creates and runs a Simulator for the given planner and problem. Extracts information from the simulator's
        trajectory.

        Args:
            planner:
                An instance of a subclass that inherits Planner, e.g. StraightLinePlanner.
            problem:
                A Problem
            sim_config:
                A YAML file for the Simulator configurations.

        Returns:
            Currently returns (success, time, list of battery_levels), but this will most likely be added to over time.
        """
        # Step 1: Set the planner's problem to the given problem
        planner.problem = problem

        # Step 2: Create and run the simulator
        sim = Simulator(planner=planner, problem=problem, project_dir=self.project_dir, sim_config=sim_config)
        success = sim.run()
        if not success:
            return False, None, (None, None, None)

        # Step 3: extract the "time" variable of the final state, which is the last element of the last list
        time = sim.trajectory[-1][-1]

        # Step 4: extract a list of all the battery levels
        battery_levels = sim.trajectory[2]

        return success, time, battery_levels

    def avg_bat_level(self, planner):
        """Finds the average battery level across all states in all problems.

        Args:
            planner:
                An instance of a subclass that inherits Planner, e.g. StraightLinePlanner.

        Returns:
            A float.
        """
        return self.general_avg_bat_level(planner=planner, aggregation_func=np.average)

    def avg_bat_level_variance(self, planner):
        """Finds the average battery level variance across all problems. The variance is calculated for each problem,
        and then averaged.

        Args:
            planner:
                An instance of a subclass that inherits Planner, e.g. StraightLinePlanner.

        Returns:
            A float.
        """
        return self.general_avg_bat_level(planner=planner, aggregation_func=np.var)

    def avg_bat_level_percent_below_threshold(self, planner, threshold=0.2):
        """Finds the average percent of time the battery level is below a given threshold across all states in all
        problems.

        Args:
            planner:
                An instance of a subclass that inherits Planner, e.g. StraightLinePlanner.

        Returns:
            A float.
        """
        def percent_below(battery_levels):
            # we will create a list of all the battery levels below the threshold to calculate percent of time below
            below_thresh = list(filter(lambda bat_level: bat_level < threshold, battery_levels))
            percent_below = len(below_thresh) / len(battery_levels)
            return percent_below

        return self.general_avg_bat_level(planner=planner, aggregation_func=percent_below)

    def general_avg_bat_level(self, planner, aggregation_func):
        """ A generic function that averages some aspect of each battery level list.

        For instance, if the aggregation_func is np.var, then this generic function finds the variance of each battery
        levels list, and returns the average of these variances.

        Args:
            planner:
                An instance of a subclass that inherits Planner, e.g. StraightLinePlanner.
            aggregation_func:
                A one argument function that returns a number, e.g. np.var.

        Returns:
            A float.
        """
        assert planner in self.all_battery_levels, "Did not run the evaluate_planner method"

        average = np.average([aggregation_func(battery_levels) for battery_levels in self.all_battery_levels[planner]])
        return average

    def avg_time(self, planner):
        """Finds the average time it takes the planner to solve a problem.

        Args:
            planner:
                An instance of a subclass that inherits Planner, e.g. StraightLinePlanner.

        Returns:
            A float.
        """
        assert planner in self.all_battery_levels, "Did not run the evaluate_planner method"

        return np.average(self.all_times[planner])

    def view_results(self, planner, ndigits=5):
        """Prints pertinent information about the planner's performance on the problems from the problem_set.

        Args:
            planner:
                An instance of a subclass that inherits Planner, e.g. StraightLinePlanner.
            ndigits:
                The number of digits to be used in rounding.

        Returns:
            None
        """
        assert planner in self.all_battery_levels, "Did not run the evaluate_planner method"

        print("-" * 50)
        print("PERCENT SUCCESSFUL: {} %".format(round(self.total_successes[planner] * 100, ndigits=ndigits)))
        print("\nFAILED PROBLEMS: ", self.failed_problems[planner])
        print("\nALL TIMES: ", self.all_times[planner])
        print("\nAVERAGE TIME: ", round(self.avg_time(planner), ndigits=ndigits))
        print("\nBATTERY LEVELS LISTS: ", self.all_battery_levels[planner])
        print("\nAVERAGE BATTERY LEVEL: ", round(self.avg_bat_level(planner), ndigits=ndigits))
        print("\nAVERAGE BATTERY VARIANCE: ", round(self.avg_bat_level_variance(planner), ndigits=ndigits))
        print("\nAVERAGE BATTERY PERCENT BELOW THRESHOLD: {} %"
              "".format(round(self.avg_bat_level_percent_below_threshold(planner, 0.2) * 100, ndigits=ndigits)))
        print("-" * 50)
