# %% Run imports

import datetime
import os

import hj_reachability as hj
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner import (
    HJBSeaweed2DPlanner,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units

# %% Initialize
os.chdir(
    "/Users/matthiaskiller/Library/Mobile Documents/com~apple~CloudDocs/Studium/Master RCI/Masters Thesis/Code/OceanPlatformControl"
)
# Initialize the Arena (holds all data sources and the platform, everything except controller)
# arena = ArenaFactory.create(scenario_name="gulf_of_mexico_HYCOM_hindcast_local_solar_seaweed")
arena = ArenaFactory.create(scenario_name="current_highway")
# we can also download the respective files directly to a temp folder, then t_interval needs to be set
# % Specify Navigation Problem
# x_0 = PlatformState(
#     lon=units.Distance(deg=-150),
#     lat=units.Distance(deg=30),
#     date_time=datetime.datetime(2022, 7, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-120), lat=units.Distance(deg=70))
x_0 = PlatformState(
    lon=units.Distance(deg=1),
    lat=units.Distance(deg=1),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=1), lat=units.Distance(deg=1))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
    timeout=datetime.timedelta(days=1),
    platform_dict=arena.platform.platform_dict,
)



# %% Plot the problem on the map


t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(), x_T=x_T, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
)



ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()

ax = arena.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()


# x, y = np.linspace(-1, 3.25,18), np.linspace(0, 3.25, 14)
# z = arena.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
# ).to_array()

# print(z)
# print("test")
# fig = go.Figure(data=[go.Surface(x=x, y=y, z=z[0][0])])

# fig.show()

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
    "T_goal_in_seconds": 3600, #* 24 * 4,
    "use_geographic_coordinate_system": False,
    "progress_bar": True,
    "grid_res": 0.1,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    "platform_dict": arena.platform.platform_dict,
    "fwd_back_buffer_in_seconds": 0,
}
planner = HJBSeaweed2DPlanner(arena=arena, problem=problem, specific_settings=specific_settings)

# %% Run reachability planner
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)

#%% get vlaue function 

planner.animate_value_func_3D()


#%%
# planner.all_values.max()
#  # %% Various plotting of the reachability computations
# planner.plot_reachability_snapshot(
#     rel_time_in_seconds=3599,
#     granularity_in_h=0.1,
#     alpha_color=1,
#     time_to_reach=False,
#     fig_size_inches=(6, 6),
#     plot_in_h=False,
# )
# planner.plot_reachability_snapshot_over_currents(rel_time_in_seconds=0, granularity_in_h=5, time_to_reach=False)
# # planner.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, filename="test_reach_animation.mp4")
# # planner.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, with_opt_ctrl=True,
# #                                   filename="test_reach_animation_w_ctrl.mp4", forward_time=True)

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
#dt_in_s = arena.platform.platform_dict["dt_in_s"]
print(arena.platform.state.seaweed_mass.kg)
# for i in tqdm(range(int(3600 * 24 * 3 / 600))):  # 3 days
for i in tqdm(range(int(200))):  #3600 * 24 * 3 / dt_in_s # 3 days

    action = planner.get_action(observation=observation)
    observation = arena.step(action)


print(arena.platform.state.seaweed_mass.kg)
arena.plot_seaweed_trajectory_on_timeaxis()


#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="seaweed")
arena.plot_all_on_map(problem=problem, background="current")

#%% Animate the trajectory
# arena.animate_trajectory(
#     problem=problem, temporal_resolution=7200, background="seaweed", output="jupyter", margin=3
# )

# %%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=7200,
    background="current",
    output="trajectory_currents.mp4",
)
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=7200,
    background="seaweed",
    output="trajectory_seaweed.mp4",
)

# # %%
# arena.plot_seaweed_trajectory_on_timeaxis()
# # %%


# #%%

# planner.plot_reachability_snapshot_over_currents(rel_time_in_seconds=1000, granularity_in_h=5, time_to_reach=False)


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
