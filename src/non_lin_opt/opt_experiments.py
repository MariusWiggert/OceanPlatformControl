import casadi as ca
from src.utils import hycom_utils, non_lin_opt_utils, simulation_utils
from src.utils.archive import particles, gif_utils
import parcels as p
import numpy as np

from src.utils import hycom_utils
from src.non_lin_opt.ipopt_planner import IpoptPlannerFixCur
import os

from src.utils.problem import Problem
from src.utils.simulator import Simulator

project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Load in data as fieldset
nc_file = 'data/' + "gulf_of_mexico_2020-11-01-10_5h.nc4"
conv_m_to_deg = 111120.
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

# Test 1 easy follow currents
x_0 = [-96.9, 22.2]
x_T = [-96.9, 22.8]

# planner fixed time horizon
T_planner = 806764

# Step 1: set up problem
problem = Problem(fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=None)
#%%
problem.viz()
#%%
u_curr_func, v_curr_func = simulation_utils.get_interpolation_func(
            problem.fieldset,
            type='bspline',
            fixed_time_index=problem.fixed_time_index)
#%%
# solve without currents for initialization
T_init = 806764.
N = 100
# T_fix = 806764.
# T_init, u_init, x_init, dt = optimal_control_utils.solve_time_opt_wo_currents(x_0, x_T, N, conv_m_to_deg, u_max, T_fix)

# Plot the results
# gif_utils.plot_opt_results(T_init, u_init, x_init, N)

#%%
# create optimization problem
opti = ca.Opti()

# declare decision variables
T = T_init
t = ca.MX.sym('t')   # time
# x, y only for now
x = opti.variable(2, N + 1)  # Decision variables for state trajetcory
u = opti.variable(2, N)
x_start = opti.parameter(2, 1)  # Parameter (not optimized over)
x_goal = opti.parameter(2, 1)  # Parameter (not optimized over)

# optimizer objective
opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))

# create the dynamics function (note here in form of u_x and u_y)
F = ca.Function('f', [t, x, u],
                [ca.vertcat(u[0] * problem.dyn_dict['u_max'] + u_curr_func(ca.vertcat(t, x[1], x[0])),
                            u[1] * problem.dyn_dict['u_max'] + v_curr_func(ca.vertcat(t, x[1], x[0])))
                 / conv_m_to_deg],
                # ,c_recharge - u[0]**2)],
                ['t', 'x', 'u'], ['x_dot'])

#%%
# add the dynamics constraints
dt = T / N
for k in range(N):
    # explicit forward euler version
    x_next = x[:, k] + dt * F(t=dt*k, x=x[:, k], u=u[:, k])['x_dot']
    opti.subject_to(x[:, k + 1] == x_next)

#%%

# boundary constraint
# opti.subject_to(T >= 0.)
# opti.subject_to(opti.bounded(0., T, 2*T_init))
opti.subject_to(x[:, 0] == x_start)
opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

# control constraints
opti.subject_to(u[0, :] ** 2 + u[1, :] ** 2 <= 1)
# opti.subject_to(opti.bounded(-u_max, u[0, :], u_max))
# opti.subject_to(opti.bounded(-u_max, u[1, :], u_max))
# opti.subject_to(opti.bounded(0, u[1, :], 2*ca.pi))
# state constraints
opti.subject_to(opti.bounded(problem.fieldset.U.grid.lon.min(),
                             x[0, :], problem.fieldset.U.grid.lon.max()))
opti.subject_to(opti.bounded(problem.fieldset.U.grid.lat.min(),
                             x[1, :], problem.fieldset.U.grid.lat.max()))
# opti.subject_to(opti.bounded(0.1, x[2, :], 1.))   # battery constraint

opti.set_value(x_start, problem.x_0[:2])
opti.set_value(x_goal, problem.x_T)
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

#%%
from src.utils import plotting_utils
plotting_utils.plot_opt_results(T/3600, u * problem.dyn_dict['u_max'], x, N)



#%%
# create optimization problem
opti = ca.Opti()

# declare decision variables
# T_init = 10000
T = T_init
# T = opti.variable()
# x, y, b, t
x = opti.variable(2, N+1)   # Decision variables for state trajetcory
u = opti.variable(2, N)
x_start = opti.parameter(2, 1)  # Parameter (not optimized over)
x_goal = opti.parameter(2, 1)  # Parameter (not optimized over)

opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))

F = ca.Function('f', [t, x, u],
                [ca.vertcat(u[0]/conv_m_to_deg + u_curr_func(ca.vertcat(x[0], x[1])),
                            u[1]/conv_m_to_deg + v_curr_func(ca.vertcat(x[0], x[1])))],
                # ,c_recharge - u[0]**2)],
                ['x', 'u'], ['x_dot'])

# add the dynamics constraints
dt = T/N
for k in range(N):
    # explicit forward euler version
    x_next = x[:, k] + dt*F(x=x[:, k], u=u[:, k])['x_dot']
    opti.subject_to(x[:, k+1] == x_next)

# boundary constraint
# opti.subject_to(T >= 0.)
# opti.subject_to(opti.bounded(0., T, 2*T_init))
opti.subject_to(x[:, 0] == x_start)
opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

# control constraints
opti.subject_to(opti.bounded(-u_max, u[0, :], u_max))
opti.subject_to(opti.bounded(-u_max, u[1, :], u_max))
# opti.subject_to(opti.bounded(0, u[1, :], 2*ca.pi))
# state constraints
opti.subject_to(opti.bounded(-98., x[0, :], -96.))
opti.subject_to(opti.bounded(21., x[1, :], 23.))
# opti.subject_to(opti.bounded(0.1, x[2, :], 1.))   # battery constraint

opti.set_value(x_start, x_0)
opti.set_value(x_goal, x_T)
# opti.set_initial(x, x_init)
# opti.set_initial(u, u_init)
opti.set_initial(T, T_init)
opti.solver('ipopt')
sol = opti.solve()

#%%
# extract the time vector and control signal
T = sol.value(T)
u = sol.value(u)
x = sol.value(x)
dt = sol.value(dt)

# T = opti.debug.value(T)
# u = opti.debug.value(u)
# x = opti.debug.value(x)
# dt = opti.debug.value(dt)
time_vec = np.arange(u.shape[1]+1) * dt
gif_utils.plot_opt_results(T, u, x, N)

#%%
# plot straight line actuation
gif_utils.gif_straight_line('trial_5_straight', x_0, x_T, T, dt, fieldset, u_max=1.)

#%%
# plot open loop control trajectory
gif_utils.gif_open_loop_control("trial_5_open_loop", x_0, u, time_vec, T, dt, fieldset, N_pictures=40)