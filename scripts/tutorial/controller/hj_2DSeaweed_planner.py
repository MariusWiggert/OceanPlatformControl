#%% Run imports
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

import wandb
from ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner import (
    HJBSeaweed2DPlanner,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.SeaweedProblem import (
    SeaweedProblem,
)
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import set_arena_loggers

# %% Initialize
os.chdir(
    "/Users/matthiaskiller/Library/Mobile Documents/com~apple~CloudDocs/Studium/Master RCI/Masters Thesis/Code/OceanPlatformControl"
)
set_arena_loggers(logging.INFO)

scenario_name = "gulf_of_mexico_HYCOM_hindcast_local_solar_seaweed"

# Initialize the Arena (holds all data sources and the platform, everything except controller)
# arena = ArenaFactory.create(scenario_name="gulf_of_mexico_HYCOM_hindcast_local_solar_seaweed")
arena = ArenaFactory.create(scenario_name=scenario_name)
# we can also download the respective files directly to a temp folder, then t_interval needs to be set
# % Specify Navigation Problem
# x_0 = PlatformState(
#     lon=units.Distance(deg=-150),
#     lat=units.Distance(deg=30),
#     date_time=datetime.datetime(2022, 7, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-120), lat=units.Distance(deg=70))

#%% init weights & biases
with open(f"config/arena/{scenario_name}.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# wandb.init(project="Long Horizon Seaweed Maximization", entity="ocean-platform-control", config=config)


#%%
# x_0 = PlatformState(
#     lon=units.Distance(deg=-83.1),
#     lat=units.Distance(deg=23.2),
#     date_time=datetime.datetime(2021, 11, 23, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-83.1), lat=units.Distance(deg=23.2))
x_0 = PlatformState(
    lon=units.Distance(deg=-80),
    lat=units.Distance(deg=-15),
    date_time=datetime.datetime(2022, 8, 23, 18, 0, tzinfo=datetime.timezone.utc),
)


problem = SeaweedProblem(
    start_state=x_0,
    platform_dict=arena.platform.platform_dict,
)


# %% Plot the problem on the map


t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(),
    x_T=x_0.to_spatio_temporal_point(),
    deg_around_x0_xT_box=1,
    temp_horizon_in_s=3600,
)

ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()

ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()

ax = arena.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()


# %% Instantiate the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 12,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "grid_res": 0.1,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    "platform_dict": arena.platform.platform_dict,
    "first_plan": True,  # indicates whether we plan the first time over the whole time-horizon or only over days with new forecast and recycle value fct. for the remaining days
    "forecast_length": 3600 * 24 * 9
    + 11 * 3600,  # length of forecast horizon -> 3600 * 24 * #days --> curr. 9 days and 11hours
}
# wandb.config.update({"planner_settings": specific_settings})

#%%
planner = HJBSeaweed2DPlanner(arena=arena, problem=problem, specific_settings=specific_settings)

# %% Run reachability planner
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)

#%% get value function #
planner.animate_value_func_3D()

#%% save planner state and reload it
# Save it to a folder
planner.save_planner_state("saved_planner/")
# Load it from the folder
# loaded_planner = HJBSeaweed2DPlanner.from_saved_planner_state(folder="saved_planner/", problem=problem, arena=arena)
# loaded_planner.last_data_source = arena.ocean_field.hindcast_data_source
# # observation = arena.reset(platform_state=x_0)
# # loaded_planner._update_current_data(observation=observation)
# # planner = loaded_planner


# %% Let the controller run closed-loop within the arena (the simulation loop)
# observation = arena.reset(platform_state=x_0)
dt_in_s = arena.platform.platform_dict["dt_in_s"]
print(arena.platform.state.seaweed_mass.kg)

for i in tqdm(range(int(3600 * 24 * 12 / dt_in_s))):
    action = planner.get_action(observation=observation)
    observation = arena.step(action)

#%%
## Seaweed growth curve
print(arena.platform.state.seaweed_mass.kg)
fig, ax = plt.subplots()
ax = arena.plot_seaweed_trajectory_on_timeaxis(ax=ax)
fig.canvas.draw()
ax.draw(fig.canvas.renderer)

# # Now we can save it to a numpy array
# data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# #image = wandb.Image(data, caption="Seaweed trajectory on timeaxis")

# #wandb.log({"seaweed_trajectory_on_timeaxis": image})

# ## Battery curve
# fig, ax = plt.subplots()
# ax = arena.plot_battery_trajectory_on_timeaxis(ax=ax)
# fig.canvas.draw()
# ax.draw(fig.canvas.renderer)

# # Now we can save it to a numpy array
# data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# #image = wandb.Image(data, caption="Battery trajectory on timeaxis")

# #wandb.log({"battery_trajectory_on_timeaxis": image})

# ## Seaweed trajectory on map
# fig, ax = plt.subplots()
# ax = arena.plot_all_on_map(problem=problem, background="seaweed",return_ax=True)
# fig.canvas.draw()
# ax.draw(fig.canvas.renderer)

# # Now we can save it to a numpy array
# data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# #image = wandb.Image(data, caption="Seaweed trajectory on map")

# #wandb.log({"seaweed_trajectory_on_map": image})

# ## Current trajectory
# fig, ax = plt.subplots()
# ax = arena.plot_all_on_map(problem=problem, background="current",return_ax=True)
# fig.canvas.draw()
# ax.draw(fig.canvas.renderer)

# # Now we can save it to a numpy array
# data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# #image = wandb.Image(data, caption="Current trajectory on map")

# #wandb.log({"current_trajectory_on_map": image})


#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="seaweed")
arena.plot_all_on_map(problem=problem, background="current")


# %%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=14400,  # 7200,
    background="current",
    output="trajectory_currents.mp4",
)
# wandb.log(
#   {"video": wandb.Video("generated_media/trajectory_currents.mp4", fps=25, format="mp4")})

arena.animate_trajectory(
    problem=problem,
    temporal_resolution=14400,  # 7200,
    background="seaweed",
    output="trajectory_seaweed.mp4",
)


#%%
wandb.log({"video": wandb.Video("generated_media/trajectory_seaweed.mp4", fps=25, format="mp4")})


# #%% Animate the trajectory
# arena.animate_trajectory(
#     problem=problem, temporal_resolution=7200, background="seaweed", output="jupyter", margin=3
# )

# # %%
# planner.vis_value_func_3D()
# # %%
# hj.viz._visSet3D(planner.grid, planner.all_values[0], level=0)
# # %%
# planner.all_values[0].ravel().shapew


# # %% Animate Value function in 3D space
# fig = go.Figure(data=go.Surface(x=planner.grid.states[..., 0],
#                             y=planner.grid.states[..., 1],
#                             z=planner.all_values[0],

# ))

# frames=[go.Frame(data=go.Surface(
#                         z=planner.all_values[k]),
#             name=str(k)) for k in range(len(planner.all_values))]
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
#                             frame= dict(duration=201, redraw=True),
#                             transition=dict(duration= 0))
#                             ],
#                         label=f'{k+1}'
#                         ) for k in range(len(planner.all_values))],
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


# # %%
# hj.viz.visSet2DAnimation(planner.grid, planner.all_values, planner.reach_times, type='gif', colorbar=False)

#%% Plotting
planner.animate_value_func_3D()

# planner.vis_value_func_3D(-1)
# planner.vis_value_func_2D(-1)
# planner.vis_value_func_contour(-1)
# %%

# proj_z=lambda x, y, z: z #projection in the z-direction
# colorsurfz=proj_z(x,y,z)
# z = planner.all_values[-1]
# z_offset=(np.min(z)-2)*np.ones(z.shape)
# fig = go.Figure(data=[go.Surface(z=list(z_offset),
#                 x=list(x),
#                 y=list(y),
#                 showlegend=False,
#                 showscale=True,
#                 surfacecolor=colorsurfz,
#                )])
# fig.show()

# fig = go.Figure(data=[go.Contour(
#         z=list(planner.all_values[-1]),
#         x=list(planner.grid.states[0]), # horizontal axis
#         y=list(planner.grid.states[1]),# vertical axis
#         contours_coloring='heatmap'
#     )])
# fig.show()

# x, y = np.linspace(-1, 3.25,18), np.linspace(0, 3.25, 14)
# z = arena.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
# ).to_array()

# print(z)
# print("test")
# fig = go.Figure(data=[go.Surface(x=x, y=y, z=z[0][0])])

# fig.show()
# %%
