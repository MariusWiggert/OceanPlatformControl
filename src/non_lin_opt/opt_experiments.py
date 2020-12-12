import casadi as ca
from src.utils import particles, hycom_utils, kernels, optimal_control_utils, gif_utils
import parcels as p
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import netCDF4

#%% Load in data as fieldset
# large field fixed cur
# nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur.nc"
# small field fixed cur
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
# nc_file = 'data/' +'gulf_of_mexico_2020-11-17_2h_var_cur.nc'
conv_m_to_deg = 111120.
u_max = 0.2  # in m/s
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

#%%
# Problem Set-Up
# # Test 1
# x_0 = [-97.4, 22.5]
# x_T = [-97.2, 21.9]

# # Test 2 around the vortex
# x_0 = [-97.4, 22.5]
# x_T = [-96.9, 22.2]

# # Test 3 long around the vortex
x_0 = [-96.9, 22.8]
x_T = [-96.9, 22.2]
# Note: can speed it up with sparsity of the hessian
N = 100

# get interpolation function
u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(fieldset, conv_m_to_deg)
#%%
pset = p.ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                             pclass=particles.TargetParticle,  # the type of particles (JITParticle or ScipyParticle)
                             lon=[x_0[0]],    # a vector of release longitudes
                             lat=[x_0[1]],   # a vector of release latitudes
                            lon_target=[x_T[0]],
                            lat_target=[x_T[1]],
                            v_max=[u_max/conv_m_to_deg])

# pset.show(field=fieldset.U)
pset.show(field='vector')
#%%
# solve without currents for initialization
T_init = 806764.
# T_fix = 806764.
# T_init, u_init, x_init, dt = optimal_control_utils.solve_time_opt_wo_currents(x_0, x_T, N, conv_m_to_deg, u_max, T_fix)

# Plot the results
# gif_utils.plot_opt_results(T_init, u_init, x_init, N)

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

opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])) + T)

F = ca.Function('f', [x, u],
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