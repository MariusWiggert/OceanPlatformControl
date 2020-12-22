import casadi as ca
import numpy as np

# solve problem without currents
def solve_time_opt_wo_currents(x_0, x_T, N, conv_m_to_deg, u_max, T_fixed = None):
    opti = ca.Opti()

    T = T_fixed  # 106764.
    # declare decision variables
    # T = opti.variable()
    # x, y
    x = opti.variable(2, N + 1)  # Decision variables for state trajetcory
    u = opti.variable(2, N)
    x_start = opti.parameter(2, 1)  # Parameter (not optimized over)
    x_goal = opti.parameter(2, 1)  # Parameter (not optimized over)

    opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))
    # specify dynamics
    F = ca.Function('f', [x, u],
                    [ca.vertcat(u[0] / conv_m_to_deg,
                                u[1] / conv_m_to_deg)],
                    # ,c_recharge - u[0]**2)],
                    ['x', 'u'], ['x_dot'])

    # add the dynamics constraints
    dt = T / N
    for k in range(N):
        # explicit forward euler version
		# # RK4 version
		# k1 = F(x=x[:, k], u=u[:,k])
		# k2 = F(x=x[:, k] + dt/2*k1['x_dot'], u=u[:, k])
		# k3 = F(x=x[:, k] + dt/2*k2['x_dot'], u=u[:, k])
		# k4 = F(x=x[:, k] + dt*k3['x_dot'], u=u[:, k])
		# x_next = x[:, k] + dt/6*(k1['x_dot']+2*k2['x_dot']+2*k3['x_dot']+k4['x_dot'])

        x_next = x[:, k] + dt * F(x=x[:, k], u=u[:, k])['x_dot']
        opti.subject_to(x[:, k + 1] == x_next)

    # boundary constraint
    # opti.subject_to(T >= 0.)
    opti.subject_to(x[:, 0] == x_start)
    opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

    # control constraints
    opti.subject_to(opti.bounded(-u_max, u[0, :], u_max))
    opti.subject_to(opti.bounded(-u_max, u[1, :], u_max))
    # opti.subject_to(opti.bounded(0, u[1, :], 2 * ca.pi))
    # state constraints
    # opti.subject_to(opti.bounded(-98., x[0, :], -96.))
    # opti.subject_to(opti.bounded(21., x[1, :], 23.))
    # opti.subject_to(opti.bounded(0.1, x[2, :], 1.))   # battery constraint

    # initialization
    x_init = np.vstack([np.linspace(x_0[0], x_T[0], N + 1), np.linspace(x_0[1], x_T[1], N + 1)])
    # u_init = np.array([[1.] * N, [3*np.pi/2] * N])
    # T_init = 20 * 60 * 60.

    opti.set_value(x_start, x_0)
    opti.set_value(x_goal, x_T)
    opti.set_initial(x, x_init)
    # opti.set_initial(u, u_init)
    # opti.set_initial(T, T_init)
    opti.solver('ipopt')
    sol = opti.solve()

    # T = sol.value(T)
    u = sol.value(u)
    x = sol.value(x)
    # dt = sol.value(dt)

    return T, u, x, dt


def get_2Dinterpolation_func(problem, type='bspline'):
    # Step 1: get grid
    xgrid = problem.fieldset.U.lon
    ygrid = problem.fieldset.U.lat
    t_grid = problem.fieldset.U.grid.time

    print("Optimizer fieldset interpolation fixed time at: {time}".format(
        time=str(problem.fieldset.U.grid.time_origin.fulltime(t_grid[problem.fixed_time_index]))))

    # Step 2: extract field data
    # [tdim, zdim, ydim, xdim]
    if len(problem.fieldset.U.data.shape) == 4:  # if there is a depth dimension in the dataset
        u_data = problem.fieldset.U.data[problem.fixed_time_index, 0, :, :]
        v_data = problem.fieldset.V.data[problem.fixed_time_index, 0, :, :]
    # [tdim, ydim, xdim]
    elif len(problem.fieldset.U.data.shape) == 3:  # if there is no depth dimension in the dataset
        u_data = problem.fieldset.U.data[problem.fixed_time_index, :, :]
        v_data = problem.fieldset.V.data[problem.fixed_time_index, :, :]

    # U field fixed
    u_curr_func = ca.interpolant('u_curr', type, [ygrid, xgrid], u_data.ravel(order='F'))
    # V field fixed
    v_curr_func = ca.interpolant('v_curr', type, [ygrid, xgrid], v_data.ravel(order='F'))

    return u_curr_func, v_curr_func
