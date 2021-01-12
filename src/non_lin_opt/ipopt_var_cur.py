from src.utils.problem import Problem
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from src.utils import hycom_utils
import os
project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
# nc_file = 'data/' + "gulf_of_mexico_2020-11-17-22_5h.nc4"
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)
u_max = 0.2     # in m/s
# Test 3 long around the vortex
x_0 = [-96.9, 22.8]
x_T = [-96.9, 22.2]

# planner fixed time horizon
T_planner = 806764

# Step 1: set up problem
prob = Problem(fieldset, x_0, x_T, u_max)
#%%
# prob.viz()
fieldset = prob.fieldset
conv_m_to_deg = 111120.
type ='bspline'
#%%

#%%
xgrid = fieldset.U.lon
ygrid = fieldset.U.lat
t_grid = fieldset.U.grid.time[:5]
n_steps = 5
# n_steps = t_grid.shape[0]
print("Fieldset from {start} to {end} in {n_steps} time steps of {time:.2f} hour(s) resolution".format(
    start=fieldset.U.grid.time_origin, end=str(fieldset.U.grid.time_origin.fulltime(t_grid[-1])),
    n_steps=n_steps, time=t_grid[1]/3600))
#%%
# U data to deg/s [tdim, zdim, ydim, xdim]
if len(fieldset.U.data.shape) == 4:     # if there is a depth dimension in the dataset
    u_data = fieldset.U.data[:, 0, :, :] / conv_m_to_deg
    v_data = fieldset.V.data[:, 0, :, :] / conv_m_to_deg

elif len(fieldset.U.data.shape) == 3:     # if there is no depth dimension in the dataset
    u_data = fieldset.U.data[:5, :, :] / conv_m_to_deg
    v_data = fieldset.V.data[:5, :, :] / conv_m_to_deg

else:
    raise NotImplementedError

#%%
# U field time_varying
# n_timesteps = 4
# t_grid = np.linspace(0, 40000, n_timesteps)
# # u_field = np.flip(u_data, 0)
# # u_field = np.array([np.flip(u_data, 0) for _ in range(n_timesteps)])
# # u_field = np.array([u_data for _ in range(n_timesteps)])
# u_field = np.stack([u_data for _ in range(n_timesteps)], axis=0)
u_curr_func = ca.interpolant('u_curr', type, [t_grid, ygrid, xgrid], u_data.ravel(order='F'))
#%%
u_func = np.zeros((t_grid.shape[0], ygrid.shape[0], xgrid.shape[0]))
for k, t in enumerate(t_grid):
    for i, x in enumerate(xgrid):
        for j, y in enumerate(ygrid):
            u_func[k, j, i] = u_curr_func([t, y, x])

#%%
plt.matshow(u_func[4,:,:])
plt.show()
#%%
plt.matshow(u_data[0,:,:])
plt.show()