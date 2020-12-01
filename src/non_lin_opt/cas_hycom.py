import casadi as ca
from utils import particles, hycom_utils, kernels, optimal_control_utils
import parcels as p
import numpy as np
import matplotlib.pyplot as plt
import glob, imageio
from datetime import timedelta

#%% Load in data as fieldset
# large field fixed cur
# nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur.nc"
# small field fixed cur
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
# nc_file = 'data/' +'gulf_of_mexico_2020-11-17_2h_var_cur.nc'
conv_m_to_deg = 111120.
u_max = 1.  # in m/s
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

#%%
# Problem Set-Up
x_0 = [-96.6, 22.8]
x_T = [-97.5, 22.2]
N = 1000


# get interpolation function
u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(fieldset, conv_m_to_deg)

#%%
# solve without currents for initialization
T_init, u_init, x_init, dt = optimal_control_utils.solve_time_opt_wo_currents(x_0, x_T, N, conv_m_to_deg)
#%%
# create optimization problem
opti = ca.Opti()

# declare decision variables
T = opti.variable()
# x, y, b, t
x = opti.variable(2, N+1)   # Decision variables for state trajetcory
u = opti.variable(2, N)
x_start = opti.parameter(2, 1)  # Parameter (not optimized over)
x_goal = opti.parameter(2, 1)  # Parameter (not optimized over)

opti.minimize(T)
# specify dynamics
F = ca.Function('f', [x, u],
                [ca.vertcat(u[0]*ca.cos(u[1])/conv_m_to_deg + u_curr_func(ca.vertcat(x[0], x[1])),
                            u[0]*ca.sin(u[1])/conv_m_to_deg + v_curr_func(ca.vertcat(x[0], x[1])))],
                # ,c_recharge - u[0]**2)],
                ['x', 'u'], ['x_dot'])

#%%


# add the dynamics constraints
dt = T/N
for k in range(N):
    # explicit forward euler version
    x_next = x[:, k] + dt*F(x=x[:, k], u=u[:, k])['x_dot']
    opti.subject_to(x[:, k+1] == x_next)

# boundary constraint
opti.subject_to(T >= 0.)
opti.subject_to(x[:, 0] == x_start)
opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

# control constraints
opti.subject_to(opti.bounded(0, u[0, :], u_max))
opti.subject_to(opti.bounded(0, u[1, :], 2*ca.pi))
# state constraints
opti.subject_to(opti.bounded(-98., x[0, :], -96.))
opti.subject_to(opti.bounded(21., x[1, :], 23.))
# opti.subject_to(opti.bounded(0.1, x[2, :], 1.))   # battery constraint

opti.set_value(x_start, x_0)
opti.set_value(x_goal, x_T)
opti.set_initial(x, x_init)
opti.set_initial(u, u_init)
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
#%%
# Plot the results
# import plotly.express as px
# def plot_opt_results(T,u,x,N):
plt.figure(1)
# plt.plot(np.linspace(0., T, N+1), x[0, :], '--')
# plt.plot(np.linspace(0., T, N+1), x[1, :], '-')
plt.plot(np.linspace(0., T, N), u[0, :], '-.')
plt.plot(np.linspace(0., T, N), u[1, :], '-.')
plt.title('Results from ipopt Optimization')
plt.xlabel('time')
plt.ylabel('value')
plt.legend(['u trajectory', 'h trajectory'])
# plt.grid()
plt.show()
#%%
# def plot_traj(x):
# Plot the results
plt.figure(1)
plt.plot(x[0, :], x[1, :], '--')
plt.title('Trajectory ipopt')
plt.xlabel('x')
plt.ylabel('y')
# plt.grid()
plt.show()

#%%

# def simulate_straight_act_baseline(fieldset, x_0, u_max):
pset = p.ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                             pclass=particles.TargetParticle,  # the type of particles (JITParticle or ScipyParticle)
                             lon=[x_0[0]],    # a vector of release longitudes
                             lat=[x_0[1]],   # a vector of release latitudes
                            lon_target=[x_T[0]],
                            lat_target=[x_T[1]],
                            v_max=[u_max/conv_m_to_deg])

# visualize as a gif
straight_line_actuation = pset.Kernel(kernels.straight_line_actuation)
N_pictures = 20.
for cnt in range(20):
    # First plot the particles
    pset.show(savefile='pics_2_gif/particles'+str(cnt).zfill(2), field='vector', land=True, vmax=2.0, show_time=0.)

    pset.execute(p.AdvectionRK4 + straight_line_actuation,  # the kernel (which defines how particles move)
                 runtime=timedelta(hours=(T/3600.)/N_pictures),  # the total length of the run
                 dt=timedelta(seconds=dt),  # the timestep of the kernel
                )

#%%
file_list = glob.glob("./pics_2_gif/*")
file_list.sort()

gif_file = './hycom_trial_straight_act_gif.gif'
with imageio.get_writer(gif_file, mode='I') as writer:
    for filename in file_list:
        image = imageio.imread(filename)
        writer.append_data(image)

# u = np.zeros(u.shape)
#%%
# create the vehicle as a particle
pset = p.ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                             pclass=particles.OpenLoopParticle,  # the type of particles (JITParticle or ScipyParticle)
                             lon=[x_0[0]],    # a vector of release longitudes
                             lat=[x_0[1]],   # a vector of release latitudes
                            control_traj=[u],
                            control_time=[time_vec],
                            v_max=[u_max/conv_m_to_deg])
#%%
# pset.show(field=fieldset.U)
# pset.show(field='vector')
open_loop_actuation = pset.Kernel(kernels.open_loop_control)
#%%
pset.execute(p.AdvectionRK4 + open_loop_actuation,  # the kernel (which defines how particles move)
                 runtime=timedelta(seconds=T),  # the total length of the run
                 dt=timedelta(seconds=dt),  # the timestep of the kernel
                )

#%%
# visualize as a gif
open_loop_actuation = pset.Kernel(kernels.open_loop_control)
N_pictures = 20.
for cnt in range(20):
    # First plot the particles
    pset.show(savefile='pics_2_gif/particles'+str(cnt).zfill(2), field='vector', land=True, vmax=2.0, show_time=0.)

    pset.execute(p.AdvectionRK4 + open_loop_actuation,  # the kernel (which defines how particles move)
                 runtime=timedelta(hours=(T/3600.)/N_pictures),  # the total length of the run
                 dt=timedelta(seconds=dt),  # the timestep of the kernel
                )
#%%
# Issue: simulation outcome is very different from the optimization x, u
x_sym_1 = ca.MX.sym('x1')
x_sym_2 = ca.MX.sym('x2')
x_sym = ca.vertcat(x_sym_1,x_sym_2)
u_sim_1 = ca.MX.sym('u_1')
u_sim_2 = ca.MX.sym('u_2')
u_sym = ca.vertcat(u_sim_1, u_sim_2)
f = ca.Function('f', [x_sym, u_sym],
                [ca.vertcat(u_sym[0]*ca.cos(u_sym[1])/conv_m_to_deg + u_curr_func(ca.vertcat(x_sym[0], x_sym[1])),
                 u_sym[0]*ca.sin(u_sym[1])/conv_m_to_deg + v_curr_func(ca.vertcat(x_sym[0], x_sym[1])))],
                ['x', 'u'], ['x_dot'])
#%%
dae = {'x': x_sym, 'p': u_sym, 'ode': f(x_sym, u_sym)}
integ = ca.integrator('F_int', 'rk', dae, {'tf': dt})

#%%
res = integ('x0', x_0, 'p', [1.00000001, 3.76859123])

#%%
file_list = glob.glob("./pics_2_gif/*")
file_list.sort()

gif_file = './hycom_trial_gif.gif'
with imageio.get_writer(gif_file, mode='I') as writer:
    for filename in file_list:
        image = imageio.imread(filename)
        writer.append_data(image)
