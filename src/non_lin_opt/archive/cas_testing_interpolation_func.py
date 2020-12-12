import casadi as ca
from src.utils import particles, hycom_utils, kernels, optimal_control_utils, gif_utils
import parcels as p
import numpy as np
import matplotlib.pyplot as plt

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
x_0 = [-97.4, 22.5]
x_T = [-97.2, 21.9]
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
T_init, u_init, x_init, dt = optimal_control_utils.solve_time_opt_wo_currents(x_0, x_T, N, conv_m_to_deg, u_max)

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

opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))

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
gif_utils.plot_opt_results(T, u, x, N)

#%%
# plot straight line actuation
gif_utils.gif_straight_line('trial_3_straight', x_0, x_T, T, dt, fieldset, u_max=1.)
#%%
# u = np.zeros(u.shape)
#%%
from datetime import timedelta

pset = p.ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                             pclass=particles.OpenLoopParticle,  # the type of particles (JITParticle or ScipyParticle)
                             lon=[x_0[0]],    # a vector of release longitudes
                             lat=[x_0[1]],   # a vector of release latitudes
                            control_traj=[u],
                            control_time=[time_vec],
                            v_max=[u_max/conv_m_to_deg]
                               )
open_loop_actuation = pset.Kernel(kernels.open_loop_control)

output_file = pset.ParticleFile(name="open_loop_control.nc", outputdt=timedelta(seconds=dt))
pset.execute(p.AdvectionRK4 + open_loop_actuation,     # the kernel (which defines how particles move)
             runtime=timedelta(seconds=T),    # the total length of the run
             dt=timedelta(seconds=dt),      # the timestep of the kernel
             output_file=output_file,
             # moviedt=timedelta(seconds=T/20), # works but fieldset needs to be time-varying!
             # movie_background_field=fieldset.U
             )

#%%
output_file.export()
file = p.plotTrajectoriesFile('open_loop_control.nc')

#%%
# extract the trajectory from output file
import netCDF4
data_netcdf4 = netCDF4.Dataset('open_loop_control.nc')

trajectory_netcdf4 = data_netcdf4.variables['trajectory'][:].data
time_netcdf4 = data_netcdf4.variables['time'][:].data[0]
lon_netcdf4 = data_netcdf4.variables['lon'][:].data[0]
lat_netcdf4 = data_netcdf4.variables['lat'][:].data[0]
#%%
# plot both
plt.figure(1)
plt.plot(lon_netcdf4, lat_netcdf4, '--')
# plt.plot(x[0, :], x[1, :], '-')
plt.title('Ocean Parcels vs. Optimizer trajectory')
plt.xlabel('x')
plt.ylabel('y')
# plt.legend(['parcels trajectory', 'optimizer trajectory'])
# plt.grid()
plt.show()
#%%
# plot open loop control trajectory
gif_utils.gif_open_loop_control("trial_3_open_loop", x_0, u, time_vec, T, dt, fieldset)
#%%
# # Issue: simulation outcome is very different from the optimization x
#%%
# Test 1: integrator with RK4 & linear interpolation function
# => does it give the same result as parcels? It should!
x_sym_1 = ca.MX.sym('x1')
x_sym_2 = ca.MX.sym('x2')
x_sym = ca.vertcat(x_sym_1, x_sym_2)
u_sim_1 = ca.MX.sym('u_1')
u_sim_2 = ca.MX.sym('u_2')
u_sym = ca.vertcat(u_sim_1, u_sim_2)

#%%
u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(fieldset, conv_m_to_deg, type='bspline')

print(u_curr_func(x_0)*conv_m_to_deg)
print(v_curr_func(x_0)*conv_m_to_deg)
print(u_curr_func(x_T)*conv_m_to_deg)
print(v_curr_func(x_T)*conv_m_to_deg)
#%%
# get the RK4 & b-spline variant
u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(fieldset, conv_m_to_deg, type='bspline')
f_rk_bspline = ca.Function('f_rk_bspline', [x_sym, u_sym],
                [ca.vertcat(u_sym[0]/conv_m_to_deg + u_curr_func(ca.vertcat(x_sym[0], x_sym[1])),
                 u_sym[1]/conv_m_to_deg + v_curr_func(ca.vertcat(x_sym[0], x_sym[1])))],
                ['x', 'u'], ['x_dot'])

dae_rk_bspline = {'x': x_sym, 'p': u_sym, 'ode': f_rk_bspline(x_sym, u_sym)}
integ_rk_bspline = ca.integrator('F_int', 'rk', dae_rk_bspline, {'tf': dt})
# now evaluate with symbols
res_rk_bspline = integ_rk_bspline(x0=x_sym, p=u_sym)
# Simplify API to (x,u)->(x_next)
F_rk_bspline = ca.Function('F_rk_bspline', [x_sym, u_sym], [res_rk_bspline['xf']], ['x','u'], ['x_next'])

# apply iteratively to get full trajectory out
sim_rk_bspline = F_rk_bspline.mapaccum(N)
x_rk_bspline = np.array(sim_rk_bspline(x_0, u))
#%%
# get the Euler & b-spline variant
# NOTE: have to implement Euler myself, only RK in the package...
# u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(fieldset, conv_m_to_deg, type='bspline')
f_ef_bspline = ca.Function('f_ef_bspline', [x_sym, u_sym],
                [ca.vertcat(u_sym[0]/conv_m_to_deg + u_curr_func(ca.vertcat(x_sym[0], x_sym[1])),
                 u_sym[1]/conv_m_to_deg + v_curr_func(ca.vertcat(x_sym[0], x_sym[1])))],
                ['x', 'u'], ['x_dot'])

F_ef_bspline= ca.Function('F_ef_bspline', [x_sym, u_sym],
                [ca.vertcat(x_sym[0] + dt*f_ef_bspline(x_sym, u_sym)[0],
                 x_sym[1] + dt*f_ef_bspline(x_sym, u_sym)[1])],
                ['x', 'u'], ['x_next'])

# apply iteratively to get full trajectory out
sim_ef_bspline = F_ef_bspline.mapaccum(N)
x_ef_bspline = np.array(sim_ef_bspline(x_0, u))
#%%
# get the Euler & linear variant
# NOTE: have to implement Euler myself, only RK in the package...
u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(fieldset, conv_m_to_deg, type='linear')
f_ef_linear = ca.Function('f_ef_linear', [x_sym, u_sym],
                [ca.vertcat(u_sym[0]/conv_m_to_deg + u_curr_func(ca.vertcat(x_sym[0], x_sym[1])),
                 u_sym[1]/conv_m_to_deg + v_curr_func(ca.vertcat(x_sym[0], x_sym[1])))],
                ['x', 'u'], ['x_dot'])

F_ef_linear = ca.Function('F_ef_linear', [x_sym, u_sym],
                [ca.vertcat(x_sym[0] + dt*f_ef_linear(x_sym, u_sym)[0],
                 x_sym[1] + dt*f_ef_linear(x_sym, u_sym)[1])],
                ['x', 'u'], ['x_next'])

# apply iteratively to get full trajectory out
sim_ef_linear = F_ef_linear.mapaccum(N)
x_ef_linear = np.array(sim_ef_linear(x_0, u))
#%%
# get the RK4 & linear variant
# NOTE: have to implement Euler myself, only RK in the package...
u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(fieldset, conv_m_to_deg, type='linear')
f_rk_linear = ca.Function('f_rk_linear', [x_sym, u_sym],
                [ca.vertcat(u_sym[0]/conv_m_to_deg + u_curr_func(ca.vertcat(x_sym[0], x_sym[1])),
                 u_sym[1]/conv_m_to_deg + v_curr_func(ca.vertcat(x_sym[0], x_sym[1])))],
                ['x', 'u'], ['x_dot'])

dae_rk_linear = {'x': x_sym, 'p': u_sym, 'ode': f_rk_linear(x_sym, u_sym)}
integ_rk_linear = ca.integrator('F_int', 'rk', dae_rk_linear, {'tf': dt})
# now evaluate with symbols
res_rk_linear = integ_rk_linear(x0=x_sym, p=u_sym)
# Simplify API to (x,u)->(x_next)
F_rk_linear = ca.Function('F_rk_linear', [x_sym, u_sym], [res_rk_linear['xf']], ['x','u'], ['x_next'])

# apply iteratively to get full trajectory out
sim_rk_linear = F_rk_linear.mapaccum(N)
x_rk_linear = np.array(sim_rk_linear(x_0, u))
#%% plot all of them together
# x_rk_linear = np.array(sim_rk_linear(x_0, u))
# x_ef_linear = np.array(sim_ef_linear(x_0, u))
# x_ef_bspline = np.array(sim_ef_bspline(x_0, u))
# x_rk_bspline = np.array(sim_rk_bspline(x_0, u))
#%%
plt.figure(1)
plt.plot(lon_netcdf4, lat_netcdf4, '--')
plt.plot(x[0, :], x[1, :], '-')
plt.plot(x_rk_linear[0, :], x_rk_linear[1, :], '-')
plt.plot(x_ef_linear[0, :], x_ef_linear[1, :], '-')
plt.plot(x_ef_bspline[0, :], x_ef_bspline[1, :], '-')
plt.plot(x_rk_bspline[0, :], x_rk_bspline[1, :], '-')
plt.title('Ocean Parcels vs. Optimizer trajectory vs. functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['parcels', 'ef_bspline Optimizer','x_rk_linear', 'x_ef_linear',
            'x_ef_bspline', 'x_rk_bspline'])
# plt.grid()
plt.show()

#%%
parcels_traj = np.vstack([lon_netcdf4, lat_netcdf4])
#%%
plt.figure(1)
# parcels_traj[:,1:-1],
for x_sim_traj in [x_rk_linear, x_ef_linear, x_rk_bspline]:
    diff_to_opt = x_ef_bspline - x_sim_traj
    norm = np.linalg.norm(diff_to_opt, axis=0)

    plt.plot(time_vec[1:], norm, '-')
# plt.plot(lon_netcdf4, lat_netcdf4, '--')
# plt.plot(x[0, :], x[1, :], '-')
# plt.plot(x_rk_linear[0, :], x_rk_linear[1, :], '-')
# plt.plot(x_ef_linear[0, :], x_ef_linear[1, :], '-')
# plt.plot(x_ef_bspline[0, :], x_ef_bspline[1, :], '-')
# plt.plot(x_rk_bspline[0, :], x_rk_bspline[1, :], '-')
# plt.title('Ocean Parcels vs. Optimizer trajectory vs. functions')
plt.xlabel('time along traj')
plt.ylabel('norm difference to x_ef_bspline traj (same as optimizer)')
plt.legend(['x_rk_linear', 'x_ef_linear', 'x_rk_bspline'])
# # plt.grid()
plt.show()
#%%
# n_timesteps = 4
type = 'linear'
xgrid = fieldset.U.lon
ygrid = fieldset.U.lat
# t_grid = np.linspace(0, 40000, n_timesteps)

# U data to deg/s
u_data = fieldset.U.data[0,0,:,:]
v_data = fieldset.V.data[0,0,:,:]

# U field fixed
# u_field = np.flip(u_data, 0)
# u_field = np.array([np.flip(u_data, 0) for _ in range(n_timesteps)])
u_curr_func = ca.interpolant('u_curr', type, [xgrid, ygrid], u_data.ravel(order='C'))
# # V field fixed
# v_field = np.flip(v_data, 0)
# v_field = np.array([np.flip(v_data, 0) for _ in range(n_timesteps)])
v_curr_func = ca.interpolant('v_curr', type, [xgrid, ygrid], v_data.ravel(order='C'))


#%%
# test if derivatives/jacobians are correct!
xgrid = fieldset.U.lon
ygrid = fieldset.U.lat

u_func = np.zeros((ygrid.shape[0], xgrid.shape[0]))
parcels_interpolate = np.zeros((ygrid.shape[0], xgrid.shape[0]))
for i, x in enumerate(xgrid):
    for j, y in enumerate(ygrid):
        # minus because we start with lower y values and then go up!
        u_func[-j, i] = u_curr_func([x, y])
        parcels_interpolate[-j, i] = fieldset.U.spatial_interpolation(ti=0, z=0, y=y, x=x, time=0.)
#%%
plt.matshow(u_func)
plt.show()

#%%
plt.matshow(parcels_interpolate)
plt.show()

#%%
plt.matshow(np.flip(fieldset.U.data[0,0,:,:],0))
plt.show()

#%%

#%%
# get values from the fieldset itself
class SampleParticle(p.ScipyParticle):
    U_sam = p.Variable('U_sam', dtype=np.float32)
    V_sam = p.Variable('V_sam', dtype=np.float32)

def SampleP(particle, fieldset, time):
    particle.U_sam = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    particle.V_sam = fieldset.V[time, particle.depth, particle.lat, particle.lon]

#%%
pset_sample = p.ParticleSet(fieldset, pclass=SampleParticle, lon=[x_T[0]], lat=[x_T[1]])
pset_sample.execute(SampleP, runtime=timedelta(seconds=1),    # the total length of the run
             dt=timedelta(seconds=dt))
pset_sample
#%%
u_0 = fieldset.U.spatial_interpolation(ti=0, z=0, y=x_0[1], x=x_0[0], time=0.)
v_0 = fieldset.V.spatial_interpolation(ti=0, z=0, y=x_0[1], x=x_0[0], time=0.)

print(u_0)
print(v_0)

#%%
# u_curr_func, v_curr_func = optimal_control_utils.get_interpolation_func(fieldset, conv_m_to_deg, type='bspline')
print(u_curr_func(x_0))
print(v_curr_func(x_0))