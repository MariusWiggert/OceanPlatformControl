from src.utils.classes import *
from src.utils.archive import gif_utils
import bisect


class IpoptPlanner(Planner):
    """ Non_linear optimizer using ipopt """
    def __init__(self, problem,
                 settings=None,
                 t_init=806764., n=100, mode='open-loop'):

        if settings is None:
            settings = {'conv_m_to_deg': 111120., 'int_pol_type': 'bspline', 'dyn_constraints': 'ef'}
        # problem containing the vector field, x_0, and x_T
        self.problem = problem

        # time to run the fixed time optimal control problem
        self.T_init = t_init
        # number of decision variables in the trajectory
        self.N = n
        self.settings = settings

        # get the current interpolation functions
        self.u_curr_func, self.v_curr_func = optimal_control_utils.get_interpolation_func(
            self.problem.fieldset, self.settings['conv_m_to_deg'], type=self.settings['int_pol_type'])

        if mode == 'open-loop':
            self.T, self.u_open_loop, self.x_solver, self.dt = self.run_optimization()
            self.control_time_vec = np.arange(self.u_open_loop.shape[1] + 1) * self.dt

    def get_next_action(self, state, rel_time):
        """ TODO: returns (thrust, header) for the next timestep """

        # an easy way of finding for each time, which index of control signal to apply
        idx = bisect.bisect_left(self.control_time_vec, rel_time)

        u_out = np.array([[self.u_open_loop[0, idx]], [self.u_open_loop[1, idx]]])*self.problem.u_max
        return u_out

    def run_optimization(self):
        # create optimization problem
        opti = ca.Opti()

        # declare decision variables
        T = self.T_init
        # x, y only for now
        x = opti.variable(2, self.N + 1)  # Decision variables for state trajetcory
        u = opti.variable(2, self.N)
        x_start = opti.parameter(2, 1)  # Parameter (not optimized over)
        x_goal = opti.parameter(2, 1)  # Parameter (not optimized over)

        # optimizer objective
        opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))

        # # get the current interpolation functions
        # u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(
        #     self.problem.fieldset, self.settings['conv_m_to_deg'], type=self.settings['int_pol_type'])

        # create the dynamics function (note here in form of u_x and u_y)
        F = ca.Function('f', [x, u],
                        [ca.vertcat(u[0]*self.problem.u_max / self.settings['conv_m_to_deg'] + self.u_curr_func(ca.vertcat(x[0], x[1])),
                                    u[1]*self.problem.u_max / self.settings['conv_m_to_deg'] + self.v_curr_func(ca.vertcat(x[0], x[1])))],
                        # ,c_recharge - u[0]**2)],
                        ['x', 'u'], ['x_dot'])

        # add the dynamics constraints
        dt = T / self.N
        for k in range(self.N):
            # explicit forward euler version
            x_next = x[:, k] + dt * F(x=x[:, k], u=u[:, k])['x_dot']
            opti.subject_to(x[:, k + 1] == x_next)

        # boundary constraint
        # opti.subject_to(T >= 0.)
        # opti.subject_to(opti.bounded(0., T, 2*T_init))
        opti.subject_to(x[:, 0] == x_start)
        opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

        # control constraints
        opti.subject_to(u[0, :]**2 + u[1, :]**2 <= 1)
        # opti.subject_to(opti.bounded(-u_max, u[0, :], u_max))
        # opti.subject_to(opti.bounded(-u_max, u[1, :], u_max))
        # opti.subject_to(opti.bounded(0, u[1, :], 2*ca.pi))
        # state constraints
        opti.subject_to(opti.bounded(self.problem.fieldset.U.grid.lon.min(),
                                     x[0, :], self.problem.fieldset.U.grid.lon.max()))
        opti.subject_to(opti.bounded(self.problem.fieldset.U.grid.lat.min(),
                                     x[1, :], self.problem.fieldset.U.grid.lat.max()))
        # opti.subject_to(opti.bounded(0.1, x[2, :], 1.))   # battery constraint

        opti.set_value(x_start, self.problem.x_0)
        opti.set_value(x_goal, self.problem.x_T)
        # opti.set_initial(x, x_init)
        # opti.set_initial(u, u_init)
        # opti.set_initial(T, T_init)
        opti.solver('ipopt')
        sol = opti.solve()

        # extract the time vector and control signal
        T = sol.value(T)
        u = sol.value(u)
        x = sol.value(x)
        dt = sol.value(dt)

        return T, u, x, dt

    def plot_opt_results(self):
        gif_utils.plot_opt_results(self.T, self.u_open_loop * self.problem.u_max, self.x_solver, self.N)
