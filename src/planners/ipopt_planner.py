from src.planners.planner import Planner
import casadi as ca
import numpy as np
from src.utils import plotting_utils, simulation_utils
import bisect


class IpoptPlanner(Planner):
    """Planner based in non-linear optimization in ipopt

        Attributes:
            t_init_in_h:
                The initial time to be used in the optimization (only fixed time Opt. implemented right now)
            n_dec_var:
                The number of decision variables in the discrete optimization problem.

            see Planner class for the rest of the attributes.
        """
    def __init__(self, problem, settings=None, t_init_in_h=224., n_dec_var=100):
        # initialize superclass
        super().__init__(problem, settings)

        # initialize values
        self.T_init = t_init_in_h*3600
        self.N = n_dec_var

        # get the current interpolation functions
        self.u_curr_func, self.v_curr_func = simulation_utils.get_interpolation_func(
            self.problem.fieldset,
            type=self.settings['int_pol_type'],
            fixed_time_index=self.problem.fixed_time_index)

        # run optimization
        self.T, self.u_open_loop, self.x_solver, self.dt = self.run_optimization()
        # create the vector containing the times when which control is active
        self.control_time_vec = np.arange(self.u_open_loop.shape[1] + 1) * self.dt

    def get_next_action(self, state):
        """When run in open loop this action is directly applied."""
        # an easy way of finding for each time, which index of control signal to apply
        idx = bisect.bisect_left(self.control_time_vec, state[3], lo=1)
        # Note: the lo=1 is because when time=0 then it would otherwise index to -1 in u_open_loop

        u_dir = np.array([[self.u_open_loop[0, idx-1]], [self.u_open_loop[1, idx-1]]])

        # transform to thrust & angle
        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        return u_out

    def run_optimization(self):
        """ Set-up the ipopt problem and solve it.

        Returns:
            - T     total time from start to end
            - u     array of controls over time (u_x, u_y)
            - x     state vector lat,lon over time
            - dt    time between to points/control times
            """
        # implemented by the child classes
        pass

    def plot_opt_results(self):
        plotting_utils.plot_opt_results(self.T/3600, self.u_open_loop * self.problem.dyn_dict['u_max'],
                                        self.x_solver, self.N)

    def get_waypoints(self):
        """Returns: a list of waypoints each containing [lon, lat, time]"""
        return np.vstack((self.x_solver, self.control_time_vec)).T.tolist()


class IpoptPlannerVarCur(IpoptPlanner):
    """ Non_linear optimizer using ipopt for current fields that are time-varying. """
    def __init__(self, problem,
                 settings=None,
                 t_init_in_h=224., n_dec_var=100):
        # initialize superclass
        super().__init__(problem, settings, t_init_in_h, n_dec_var)

    def run_optimization(self):
        # create optimization problem
        opti = ca.Opti()

        # declare fixed End time and variable t for indexing to variable current
        T = self.T_init
        t = ca.MX.sym('t')  # time symbolic variable

        # declare decision variables
        # For now 2 State system (X, Y), not capturing battery or seaweed mass
        x = opti.variable(2, self.N + 1)  # Decision variables for state trajetcory
        u = opti.variable(2, self.N)

        # Parameters (not optimized over)
        x_start = opti.parameter(2, 1)
        x_goal = opti.parameter(2, 1)

        # optimizer objective
        opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))

        # create the dynamics function (note here in form of u_x and u_y)
        F = ca.Function('f', [t, x, u],
                        [ca.vertcat(u[0]*self.problem.dyn_dict['u_max'] + self.u_curr_func(ca.vertcat(t, x[1], x[0])),
                                    u[1]*self.problem.dyn_dict['u_max'] + self.v_curr_func(ca.vertcat(t, x[1], x[0])))
                         / self.settings['conv_m_to_deg']],
                        # ,c_recharge - u[0]**2)],
                        ['t', 'x', 'u'], ['x_dot'])

        # add the dynamics constraints
        dt = T / self.N
        for k in range(self.N):
            # explicit forward euler version
            x_next = x[:, k] + dt * F(t=dt * k, x=x[:, k], u=u[:, k])['x_dot']
            opti.subject_to(x[:, k + 1] == x_next)

        # start state & goal constraint
        opti.subject_to(x[:, 0] == x_start)
        opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

        # control constraints
        opti.subject_to(u[0, :]**2 + u[1, :]**2 <= 1)

        # State constraints to the map boundaries
        opti.subject_to(opti.bounded(self.problem.fieldset.U.grid.lon.min(),
                                     x[0, :], self.problem.fieldset.U.grid.lon.max()))
        opti.subject_to(opti.bounded(self.problem.fieldset.U.grid.lat.min(),
                                     x[1, :], self.problem.fieldset.U.grid.lat.max()))

        # battery constraint
        # opti.subject_to(opti.bounded(0.1, x[2, :], 1.))

        # Set the values for the optimization problem
        opti.set_value(x_start, self.problem.x_0[:2])
        opti.set_value(x_goal, self.problem.x_T)

        # Optional: initialize variables for the optimizer
        # opti.set_initial(x, x_init)
        # opti.set_initial(u, u_init)

        opti.solver('ipopt')
        sol = opti.solve()

        # extract the time vector and control signal
        T = sol.value(T)
        u = sol.value(u)
        x = sol.value(x)
        dt = sol.value(dt)

        return T, u, x, dt


class IpoptPlannerFixCur(IpoptPlanner):
    """ Non_linear optimizer using ipopt for current fields that are fixed. """
    def __init__(self, problem, settings=None, t_init_in_h=806764., n_dec_var=100):
        # initialize superclass
        super().__init__(problem, settings, t_init_in_h, n_dec_var)

    def run_optimization(self):
        # create optimization problem
        opti = ca.Opti()

        # fixed end_time
        T = self.T_init

        # declare decision variables
        # For now 2 State system (X, Y), not capturing battery or seaweed mass
        x = opti.variable(2, self.N + 1)  # Decision variables for state trajetcory
        u = opti.variable(2, self.N)

        # Parameters (not optimized over)
        x_start = opti.parameter(2, 1)
        x_goal = opti.parameter(2, 1)

        # optimizer objective (minimize actuation effort)
        opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))

        # create the dynamics function
        # Note: using u_x and u_y instead of thrust & heading for smoother optimization landscape
        F = ca.Function('f', [x, u],
                        [ca.vertcat(u[0]*self.problem.dyn_dict['u_max'] + self.u_curr_func(ca.vertcat(x[1], x[0])),
                                    u[1]*self.problem.dyn_dict['u_max'] + self.v_curr_func(ca.vertcat(x[1], x[0])))
                         / self.settings['conv_m_to_deg']],
                        # ,c_recharge - u[0]**2)],
                        ['x', 'u'], ['x_dot'])

        # add the dynamics constraints
        dt = T / self.N
        for k in range(self.N):
            # explicit forward euler version
            x_next = x[:, k] + dt * F(x=x[:, k], u=u[:, k])['x_dot']
            opti.subject_to(x[:, k + 1] == x_next)

        # Start_state & goal constraint
        opti.subject_to(x[:, 0] == x_start)
        opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

        # control magnitude constraints
        opti.subject_to(u[0, :]**2 + u[1, :]**2 <= 1)

        # State constraints to the map boundaries
        opti.subject_to(opti.bounded(self.problem.fieldset.U.grid.lon.min(),
                                     x[0, :], self.problem.fieldset.U.grid.lon.max()))
        opti.subject_to(opti.bounded(self.problem.fieldset.U.grid.lat.min(),
                                     x[1, :], self.problem.fieldset.U.grid.lat.max()))

        # battery constraint
        # opti.subject_to(opti.bounded(0.1, x[2, :], 1.))

        # Set the values for the optimization problem
        opti.set_value(x_start, self.problem.x_0[:2])
        opti.set_value(x_goal, self.problem.x_T)

        # Optional: initialize variables for the optimizer
        # opti.set_initial(x, x_init)
        # opti.set_initial(u, u_init)

        opti.solver('ipopt')
        sol = opti.solve()

        # extract the time vector and control signal
        T = sol.value(T)
        u = sol.value(u)
        x = sol.value(x)
        dt = sol.value(dt)

        return T, u, x, dt