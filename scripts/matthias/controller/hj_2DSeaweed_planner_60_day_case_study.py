#%% Run imports
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
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
from ocean_navigation_simulator.ocean_observer.NoObserver import NoObserver
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import get_c3, set_arena_loggers
from ocean_navigation_simulator.utils.plotting_utils import set_palatino_font, set_palatino_font_plotly
set_palatino_font("ocean_navigation_simulator/package_data/font/Palatino_thin.ttf")
set_palatino_font_plotly("ocean_navigation_simulator/package_data/font/Palatino_thin.ttf")
os.close(0)  # close standard input stream
os.close(1)  # close standard output stream
os.close(2)  # close standard error stream
os.open(os.devnull, os.O_RDWR)  # open null device as standard input stream
os.dup(0)  # duplicate standard input stream to standard output stream

%load_ext autoreload
%autoreload 2
# %% Initialize
# os.chdir(
#     "/Users/matthiaskiller/Library/Mobile Documents/com~apple~CloudDocs/Studium/Master RCI/Masters Thesis/Code/OceanPlatformControl"
# )
set_arena_loggers(logging.ERROR)

scenario_name = "Region_M_HC_as_forecast_daily+monthly_averages_Copernicus_HC_solar_seaweed_60_day_study"  # "Region_3_Copernicus_HC_solar_seaweed"

# Initialize the Arena (holds all data sources and the platform, everything except controller)
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
    lon=units.Distance(deg=-91),
    lat=units.Distance(deg=-3.05),
    date_time=datetime.datetime(2022, 1, 4, 16, 16, 36, tzinfo=datetime.timezone.utc),
)
# %%
arena = ArenaFactory.create(
    scenario_name=scenario_name,
    # t_interval=[
    #     x_0.date_time - datetime.timedelta(days=2),
    #     x_0.date_time + datetime.timedelta(days=65),
    # ],
    points=[x_0.to_spatial_point()],
)

problem = SeaweedProblem(
    start_state=x_0,
    platform_dict=arena.platform.platform_dict,
)

# %% arena

# %% Plot the problem on the map


t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(),
    x_T=x_0.to_spatio_temporal_point(),
    deg_around_x0_xT_box=12,
    temp_horizon_in_s=3600,
)
lat_bnds = [-40, 0]
lon_bnds = [-130, -70]
# # %%
ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time,
    x_interval=lon_bnds,
    y_interval=lat_bnds,
    return_ax=True,
    quiver_spatial_res=1,
    quiver_scale=22,
    # vmax=1.2,
    alpha=0,
    quiver_headwidth=1.2,
)
# problem.plot(ax=ax)
#plt.tight_layout()
plt.savefig("currents_25S_0_108W_70W.png", dpi=1200)
plt.show()
# # %%
ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()
# %%
ax = arena.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
#%%
fig = ax.get_figure()
fig = plt.figure(fig)

# plt.savefig("seaweed_trajectory_on_map.svg")
problem.plot(ax=ax)
plt.show()


# %% Instantiate the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 24 * 60,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 60,  # area over which to run HJ_reachability
    "deg_around_xt_xT_box_average": 12,  # area over which to run HJ_reachability for average data
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 60 - 1,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "grid_res": 0.166,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "r": 0.166,  # Grid res for average data  Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    "calc_opt_traj_after_planning": False,
    "x_interval_seaweed": [-130, -70],
    "y_interval_seaweed": [-40, 0],
    "dirichlet_boundry_constant": 1,
    "discount_factor_tau": False,# 3600 * 24 * 80 - 1,  # 50 #False, #10,
    "affine_dynamics": True,
}
# wandb.config.update({"planner_settings": specific_settings})
#%%
planner = HJBSeaweed2DPlanner(arena=arena, problem=problem, specific_settings=specific_settings)
observer = NoObserver()
# %% Run reachability planner
observation = arena.reset(platform_state=x_0)
observer.observe(observation)
observation.forecast_data_source = observer
action = planner.get_action(observation=observation)
#%% get value function #
planner.animate_value_func_3D()
#%%
planner.vis_value_func_3D(0)
# %% Let the controller run closed-loop within the arena (the simulation loop)
# observation = arena.reset(platform_state=x_0)
dt_in_s = arena.platform.platform_dict["dt_in_s"]
print(arena.platform.state.seaweed_mass.kg)

problem_status = arena.problem_status(problem=problem)

while problem_status == 0:
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
    observer.observe(observation)
    observation.forecast_data_source = observer

    # update problem status
    problem_status = arena.problem_status(problem=problem)

#%%
## Seaweed growth curve
print(arena.platform.state.seaweed_mass.kg)
fig, ax = plt.subplots()
ax = arena.plot_seaweed_trajectory_on_timeaxis(ax=ax)
fig.canvas.draw()
ax.draw(fig.canvas.renderer)

plt.savefig("seaweed_trajectory_on_timeaxis_DF.png", dpi=1200)
#%%
ax = arena.plot_all_on_map(
    problem=problem,
    background="current",
    quiver_spatial_res=1.2,
    quiver_scale=19,
    x_interval=lon_bnds,
    y_interval=lat_bnds,
    show_control_trajectory=False,
    #stride=50,
    quiver_headwidth=1,
    return_ax=True,
)
#ax.get_legend().remove()
fig = ax.get_figure()

plt.show()


#%%

ax = arena.plot_all_on_map(
     problem=problem, background="seaweed", x_interval=[-105,-75], y_interval=[-25,0],
    show_control_trajectory=False,
    return_ax=True,
)
ax.get_legend().remove()
fig = ax.get_figure()
plt.savefig("seaweed_trajectory_on_map_DF_81.png", dpi=1200)
plt.show()



# %%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=14400,  # 7200,
    background="current",
    quiver_spatial_res=1.8,
    quiver_scale=15,
    # margin=8,
    # x_interval=[-98,-72],
    # y_interval=[0,-9],
    output="trajectory_currents_greedy5_60_4_.mp4",
)
# # wandb.log(
# #   {"video": wandb.Video("generated_media/trajectory_currents.mp4", fps=25, format="mp4")})
# # %%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=14400,  # 7200,
    background="seaweed",
    # margin=8,
    # x_interval=[-98,-72],
    # y_interval=[0,-9],
    output="trajectory_seaweed_greedy5_60_4.mp4",
)

ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time,
    x_interval=lon_bnds,
    y_interval=lat_bnds,
    return_ax=True,
    quiver_spatial_res=0.65,
    quiver_scale=25,
    vmax=0.7,
    alpha=0,
    quiver_headwidth=1.2,
)

#%%
hj_values = planner.all_values
posix_times = planner.reach_times + planner.current_data_t_0
x_grid = planner.grid.coordinate_vectors[0]
y_grid = planner.grid.coordinate_vectors[1]

data = xr.Dataset(
    {"HJValueFunc": (("time", "x", "y"), hj_values)},
    coords={"x": x_grid, "y": y_grid, "time": posix_times},
)

# Save to file
end_time = problem.start_state.date_time + datetime.timedelta(
    seconds=specific_settings["T_goal_in_seconds"]
)

nc_file_name = f'HJ_func_{problem.start_state.date_time.strftime("%Y-%m-%dT%H-%M-%SZ")}-{end_time.strftime("%Y-%m-%dT%H-%M-%SZ")}_umax_{config["platform_dict"]["u_max_in_mps"]}.nc'

data.to_netcdf(nc_file_name)
# %%
data = xr.open_dataset("HJ_func_2022-01-04T16-16-36Z-2022-03-05T16-16-35Z_umax_0.3.nc")
data
# %%
