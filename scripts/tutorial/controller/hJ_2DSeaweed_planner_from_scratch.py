# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio

from ocean_navigation_simulator.controllers.hj_planners.Platform2dSeaweedForSim import (
    Platform2dSeaweedForSim,
)

pio.renderers.default = "browser"
import time

import hj_reachability as hj
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from scipy.interpolate import interp1d

times = np.linspace(0, 60, 1000)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(lo=np.array([-5.0, -5.0]), hi=np.array([5.0, 5.0])), (100, 100)
)
initial_values = jnp.zeros(grid.shape)
solver_settings = hj.SolverSettings.with_accuracy(
    accuracy="high",
    artificial_dissipation_scheme=hj.artificial_dissipation.local_local_lax_friedrichs,
)
Plat2D = Platform2dSeaweedForSim(
    u_max=0.5,
    d_max=0,
    use_geographic_coordinate_system=False,
    control_mode="min",
    disturbance_mode="max",
)

#%%
# T, y, x
water_u = np.zeros((100, 100, 100))
water_v = np.zeros((100, 100, 100))
seaweed = np.zeros((100, 100, 100))
seaweed[:, 40:60, 40:60] = 1


lon = np.linspace(-5, 5, 100)
lat = np.linspace(-5, 5, 100)
time = np.linspace(0, 60, 100)

ds_seaweed = xr.Dataset(
    data_vars=dict(
        F_NGR_per_second=(["relative_time", "lat", "lon"], seaweed),
    ),
    coords=dict(
        lon=lon,
        lat=lat,
        relative_time=time,
    ),
)

ds_water = xr.Dataset(
    data_vars=dict(
        water_u=(["relative_time", "lat", "lon"], water_u),
        water_v=(["relative_time", "lat", "lon"], water_v),
    ),
    coords=dict(
        lon=lon,
        lat=lat,
        relative_time=time,
    ),
)

Plat2D.update_jax_interpolant(ds_water, ds_seaweed)

#%%
# interpolated = np.zeros((100,100))
# for i in range(grid.states.shape[0]):
#     for j in range(grid.states.shape[1]):
#         interpolated[i,j] = Plat2D.seaweed_rate(state=grid.states[i,j], time=0)
# #Plat2D._get_seaweed_growth_rate(state=, time=)
# #%%
# plt.imshow(interpolated)

# %%

plt.imshow(ds_seaweed["F_NGR_per_second"].fillna(0).data[0, ...])

# #%%
# plt.imshow(ds_water["water_v"].fillna(0).data[0,...])

# #%%
# plt.imshow(ds_water["water_u"].fillna(0).data[0,...])


#%%

times, all_values = hj.solve(solver_settings, Plat2D, grid, times, initial_values)
#%% viz
# hj.viz.visSet2DAnimation(grid, all_values, times, type='mp4', colorbar=False)
#%%


# Read data from a csv

z = all_values[-1]
x = grid.states[..., 0]
y = grid.states[..., 1]
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
fig.show()


x_init = np.array([10, 10])

# %%
print(type(grid.domain.lo))
print(type(x_init))

# %% Extract trajectory
times, x_traj, contr_seq, distr_seq = Plat2D.backtrack_trajectory(grid, x_init, times, all_values)
# %% Plot the Traj
hj.viz.visTrajSet2D(
    x_traj=x_traj,
    grid=grid,
    all_values=all_values,
    times=times,
    x_init=x_init,
)

# %% Plot the Control Traj
plt.plot(np.arange(contr_seq.shape[1]), contr_seq[0, :], color="r")
plt.plot(np.arange(contr_seq.shape[1]), contr_seq[1, :], color="b")
plt.show()


# %% RUN BACKWARD
times = np.linspace(0, 60, 1000)
times = np.flip(times, axis=0)

times, all_values = hj.solve(solver_settings, Plat2D, grid, times, all_values[-1])
#%% viz
# hj.viz.visSet2DAnimation(grid, all_values, times, type='mp4', colorbar=False)
#%%

# Read data from a csv

z = all_values[-1]
x = grid.states[..., 0]
y = grid.states[..., 1]
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
fig.show()

x_init = np.array([10, 10])

# %%
print(type(grid.domain.lo))
print(type(x_init))

# %% Extract trajectory
times, x_traj, contr_seq, distr_seq = Plat2D.backtrack_trajectory(grid, x_init, times, all_values)
# %% Plot the Traj
hj.viz.visTrajSet2D(
    x_traj=x_traj,
    grid=grid,
    all_values=all_values,
    times=times,
    x_init=x_init,
)

# %% Plot the Control Traj
plt.plot(np.arange(contr_seq.shape[1]), contr_seq[0, :], color="r")
plt.plot(np.arange(contr_seq.shape[1]), contr_seq[1, :], color="b")
plt.show()
