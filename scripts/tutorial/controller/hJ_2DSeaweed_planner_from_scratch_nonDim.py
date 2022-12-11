# %%
import time

import hj_reachability as hj
import jax.numpy as jnp

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr

from ocean_navigation_simulator.controllers.hj_planners.Platform2dSeaweedForSim import (
    Platform2dSeaweedForSim,
)

# %%
times = np.linspace(0, 40, 2000)

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(lo=np.array([-5.0, -5.0]), hi=np.array([5.0, 5.0])), (100, 100)
)

Plat2D = Platform2dSeaweedForSim(
    u_max=0.5,
    d_max=0,
    use_geographic_coordinate_system=False,
    control_mode="min",
    disturbance_mode="max",
)

Plat2D_nonDim = hj.dynamics.NonDimDynamics(Plat2D)

# extract the characteristic scale and offset value for each dimensions
characteristic_vec = grid.domain.hi - grid.domain.lo
offset_vec = grid.domain.lo

grid_nonDim = hj.Grid.nondim_grid_from_dim_grid(
    dim_grid=grid,
    characteristic_vec=characteristic_vec,
    offset_vec=offset_vec,
)

Plat2D_nonDim.characteristic_vec = characteristic_vec
Plat2D_nonDim.offset_vec = offset_vec

Plat2D_nonDim.dimensional_dynamics.characteristic_vec = characteristic_vec
Plat2D_nonDim.dimensional_dynamics.offset_vec = offset_vec


initial_values = jnp.zeros(grid_nonDim.shape)
solver_settings = hj.SolverSettings.with_accuracy(
    accuracy="high",
    artificial_dissipation_scheme=hj.artificial_dissipation.local_local_lax_friedrichs,
)

# Plat2D_nonDim.dimensional_dynamics.control_mode = "max"
# Plat2D_nonDim.dimensional_dynamics.disturbance_mode = "min"


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

Plat2D_nonDim.dimensional_dynamics.update_jax_interpolant(ds_water, ds_seaweed)

#%%
# def get_dim_state(state_nonDim: jnp.ndarray):
#         """Returns the state transformed from non_dimensional coordinates to dimensional coordinates."""
#         return state_nonDim * characteristic_vec + offset_vec

# interpolated = np.zeros((100,100))
# for i in range(grid_nonDim.states.shape[0]):
#     for j in range(grid_nonDim.states.shape[1]):
#         interpolated[i,j] = Plat2D_nonDim.dimensional_dynamics.seaweed_rate(state=get_dim_state(grid_nonDim.states[i,j]), time=99)


# Plat2D._get_seaweed_growth_rate(state=, time=)
# %%
# plt.imshow(interpolated)

# # %%

# plt.imshow(ds_seaweed["F_NGR_per_second"].fillna(0).data[0,...])

# #%%
# plt.imshow(ds_water["water_v"].fillna(0).data[0,...])

# #%%
# plt.imshow(ds_water["water_u"].fillna(0).data[0,...])


#%%
# initial_values= initial_values - 5


times, all_values = hj.solve(solver_settings, Plat2D_nonDim, grid_nonDim, times, initial_values)
#%% viz
# hj.viz.visSet2DAnimation(grid_nonDim, all_values, times, type='mp4', colorbar=False)
#%%
z = all_values[-1]
x = grid_nonDim.states[..., 0]
y = grid_nonDim.states[..., 1]
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
fig.show()


# # %% Extract trajectory
# times, x_traj, contr_seq, distr_seq = Plat2D.backtrack_trajectory(grid, x_init, times, all_values)
# # %% Plot the Traj
# hj.viz.visTrajSet2D(x_traj=x_traj, grid=grid, all_values=all_values, times=times, x_init=x_init)
# # %% Plot the Control Traj
# plt.plot(np.arange(contr_seq.shape[1]), contr_seq[0, :], color='r')
# plt.plot(np.arange(contr_seq.shape[1]), contr_seq[1, :], color='b')
# plt.show()
# %%
# %% Animate Value function in 3D space
# fig = go.Figure(data=go.Surface(x=grid.states[..., 0],
#                             y=grid.states[..., 1],
#                             z=all_values[0],

# ))

# frames=[go.Frame(data=go.Surface(
#                         z=all_values[k]),
#             name=str(k)) for k in range(len(all_values))]
# updatemenus = [dict(
#     buttons = [
#         dict(
#             args = [None, {"frame": {"duration": 20, "redraw": True},
#                             "fromcurrent": True, "transition": {"duration": 0}}],
#             label = "Play",
#             method = "animate"
#             ),
#         dict(
#             args = [[None], {"frame": {"duration": 0, "redraw": False},
#                             "mode": "immediate",
#                             "transition": {"duration": 0}}],
#             label = "Pause",
#             method = "animate"
#             )
#     ],
#     direction = "left",
#     pad = {"r": 10, "t": 87},
#     showactive = False,
#     type = "buttons",
#     x = 0.21,
#     xanchor = "right",
#     y = -0.075,
#     yanchor = "top"
# )]

# sliders = [dict(steps = [dict(method= 'animate',
#                         args= [[f'{k}'],
#                         dict(mode= 'immediate',
#                             frame= dict(duration=1000, redraw=True),
#                             transition=dict(duration= 0))
#                             ],
#                         label=f'{k+1}'
#                         ) for k in range(len(all_values))],
#             active=0,
#             transition= dict(duration= 0 ),
#             x=0, # slider starting position
#             y=0,
#             currentvalue=dict(font=dict(size=12),
#                             prefix='frame: ',
#                             visible=True,
#                             xanchor= 'center'
#                             ),
#             len=1.0) #slider length
#     ]

# fig.update_layout(width=700, height=700, updatemenus=updatemenus, sliders=sliders)
# fig.update(frames=frames)
# fig.update_traces(showscale=False)
# fig.show()
#%%
x_init = np.array([10, 10])

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
hj.viz.visTrajSet2D(x_traj=x_traj, grid=grid, all_values=all_values, times=times, x_init=x_init)

# %% Plot the Control Traj
plt.plot(np.arange(contr_seq.shape[1]), contr_seq[0, :], color="r")
plt.plot(np.arange(contr_seq.shape[1]), contr_seq[1, :], color="b")
plt.show()
# %%
