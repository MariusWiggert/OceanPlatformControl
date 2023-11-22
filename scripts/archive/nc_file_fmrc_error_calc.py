import datetime
import logging
import os

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.PlatformState import (
    SpatioTemporalPoint, SpatialPoint,
)
from ocean_navigation_simulator.problem_factories.Constructor import Constructor
from ocean_navigation_simulator.utils.misc import set_arena_loggers

%load_ext autoreload
%autoreload 2

# Set-Up for Simulation
#% Platform Speed, Forecasted and true currents
# Forecast System that we can use:
# - HYCOM Global (daily forecasts, xh resolution, 1/12 deg spatial resolution)
# - Copernicus Global (daily forecasts for 5 days out, xh resolution, 1/12 deg spatial resolution)
# - NOAA West Coast Nowcast System (daily nowcasts for 24h, xh resolution, 10km spatial resolution)
# Note: if you switch it, you need to delete the old FC files, otherwise the system will use those. It just grabs all the files in the folder 'folder_for_forecast_files'

max_speed_of_platform_in_meter_per_second = 0.1
time_horizon_in_day = 1
forecast_system_to_use = "noaa"  # either of ['HYCOM', 'Copernicus', 'noaa']

folder_for_forecast_files = "/Volumes/Data/2_Work/2_Graduate_Research/1_Seaweed/jupyter_FC_data/noaa_forecast_files/"
folder_for_hindcast_files = "/Volumes/Data/2_Work/2_Graduate_Research/1_Seaweed/cop_hc_seaweed_cali/"
#%%
for fold_path in [folder_for_hindcast_files, folder_for_forecast_files]:
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
#%
true_ocean_current_dict = {
            "field": "OceanCurrents",
            "source": "hindcast_files",
            "source_settings": {"source": "HYCOM",
                                "folder": folder_for_hindcast_files,
                                "type": "hindcast"},
        }

# Configs for simulator
simulation_timeout = 3600*24*time_horizon_in_day # 5 days

arena_config = {
    "timeout": simulation_timeout,
    "casadi_cache_dict": {
        "deg_around_x_t": 0.8,
        "time_around_x_t": 3600*24*4,
    },  # This is 24h in seconds!
    "platform_dict": {
        "battery_cap_in_wh": 400000.0, #400000.0
        "u_max_in_mps": max_speed_of_platform_in_meter_per_second,
        "motor_efficiency": 1.0,
        "solar_panel_size": 100,
        "solar_efficiency": 1.0,
        "drag_factor": 0.1,
        "dt_in_s": 600.0,
    },
    "use_geographic_coordinate_system": True,
    "spatial_boundary": None,
    "ocean_dict": {
        "region": "Region 1",  # This is the region of northern California
        "hindcast": true_ocean_current_dict,
        "forecast": #None,#true_ocean_current_dict,
            {
            "field": "OceanCurrents",
            "source": "forecast_files",
            "source_settings": {"source": forecast_system_to_use,
                                "folder": folder_for_forecast_files,
                                "type": "forecast"},
        },
    },
    "solar_dict": {"hindcast": {
        'field': 'SolarIrradiance',
        'source': 'analytical_wo_caching',  # can also be analytical_w_caching
        'source_settings': {
            'boundary_buffer': [0.2, 0.2],
            'x_domain': [-180, 180],
            'y_domain': [-90, 90],
            'temporal_domain': ["2022-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"],
            'spatial_resolution': 0.25,
            'temporal_resolution': 3600}
        }, "forecast": None},
    "seaweed_dict": {
        # "hindcast": {
        # 'field': 'SeaweedGrowth',
        # 'source': 'SeaweedGrowthCircles',
        # 'source_settings': {
        #     # Specific for it
        #     'cirles': [[-123, 37, 0.3]], # [x, y, r]
        #     'NGF_in_time_units': [0.000001], # [NGF]
        #     # Just boundary stuff
        #     'boundary_buffers': [0., 0.],
        #     'x_domain': [-130, -120],
        #     'y_domain': [35,40],
        #     'temporal_domain': ["2022-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"],
        #     'spatial_resolution': 0.1,
        #     'temporal_resolution': 3600*24}}
        # "hindcast": {
        # 'field': 'SeaweedGrowth',
        # 'source':'GEOMAR',
        # 'source_settings':{
        #     'filepath': './ocean_navigation_simulator/package_data/nutrients/'}  # './data/nutrients/2022_monthly_nutrients_and_temp.nc'
        # }
        "hindcast": {
        'field': 'SeaweedGrowth',
        'source': 'California',
        'source_settings': {
            'filepath': './ocean_navigation_simulator/package_data/cali_growth_map/static_growth_map_south.nc',
            'max_growth': 0.4,
            'respiration_rate': 0.01},
        }
        , "forecast": None},
    "bathymetry_dict" : {
        "field": "Bathymetry",
        "source": "gebco",
        "source_settings": {
            "filepath": "bathymetry_global_res_0.083_0.083_max.nc"
        },
        "distance": {
            "filepath": "bathymetry_distance_res_0.004_max_elevation_0_northern_california.nc",
            "safe_distance": 0.01, # ≈4km from shore
        },
        "casadi_cache_settings": {"deg_around_x_t": 20}, # is used by Bathymetry source casadi, not sure why necessary.
    }
}

objectiveConfig = {"type": "max_seaweed"}

#% Controller Settings
t_max_planning_ahead_in_seconds = 3600*24*2 # that is 60h

# distance controller should keep from shore
safe_distance = 0.1, # ≈4km from shore
dirichlet_and_obstacle_val = 0.4
days_ahead = time_horizon_in_day
specific_settings = {
    'ctrl_name': 'ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner.HJBSeaweed2DPlanner',
    "replan_on_new_fmrc": True,
    "direction": "backward",
    "n_time_vector": 24 * days_ahead,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1,  # area over which to run HJ_reachability
    "deg_around_xt_xT_box_average": 1,
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * days_ahead,
    "use_geographic_coordinate_system": True,
    'obstacle_dict': {'obstacle_value': dirichlet_and_obstacle_val,
                    'obstacle_file': 'bathymetry_distance_res_0.004_max_elevation_0_northern_california.nc',
                    'safe_distance_to_obstacle': safe_distance},
    "progress_bar": True,
    "grid_res": 0.04,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "calc_opt_traj_after_planning": False,
    # "x_interval_seaweed": [-130, -70],
    # "y_interval_seaweed": [-40, 0],
    "dirichlet_boundry_constant": dirichlet_and_obstacle_val,
    "discount_factor_tau": False,# 3600 * 24 * 80 - 1,  # 50 #False, #10,
    "affine_dynamics": True,
}

#% Mission Settings Start -> Target
print("current UTC datetime is: ", datetime.datetime.now())

start_time = "2023-10-05T19:00:00+00:00"  # in UTC
OB_point = {"lat": 37.738160, "lon": -122.545469}
HMB_point = {"lat": 37.482812, "lon": -122.531450}
open_ocean = {"lat": 37, "lon": -123}

x_0_dict = {"date_time": start_time,
            "battery_charge": arena_config["platform_dict"]["battery_cap_in_wh"]*3600}
x_0_dict.update(HMB_point)
missionConfig = {
    "target_radius": 0.02,  # in degrees
    "x_0": [x_0_dict],  # the start position and time
    "x_T": OB_point,  # the target position
}

#%% Download FC Files (does not need to be rerun)
# Note: This needs to be run only once, not when re-running constructor etc.
point_to_check = SpatioTemporalPoint.from_dict(missionConfig['x_0'][0])
t_interval = [point_to_check.date_time - datetime.timedelta(hours=35),
              point_to_check.date_time + datetime.timedelta(
                  seconds=simulation_timeout
                          + arena_config['casadi_cache_dict'][
                      'time_around_x_t'] + 7200)]


#HC Note: it connects to C3 and downloads the relevant forecast files to the local repo
ArenaFactory.download_required_files(
        archive_source = arena_config['ocean_dict']['hindcast']['source_settings']['source'],
        archive_type = arena_config['ocean_dict']['hindcast']['source_settings']['type'],
        download_folder=arena_config['ocean_dict']['hindcast']['source_settings']['folder'],
        t_interval = t_interval,
        region= arena_config['ocean_dict']['region'],
        points= [point_to_check])

#%FC Note: it connects to C3 and downloads the relevant forecast files to the local repo
ArenaFactory.download_required_files(
        archive_source = arena_config['ocean_dict']['forecast']['source_settings']['source'],
        archive_type = arena_config['ocean_dict']['forecast']['source_settings']['type'],
        download_folder=arena_config['ocean_dict']['forecast']['source_settings']['folder'],
        t_interval = t_interval,
        region= arena_config['ocean_dict']['region'],
        points= [point_to_check])

#%% Create Arena, Problem, Controller
set_arena_loggers(logging.DEBUG)
# Step 0: Create Constructor object which contains arena, problem, controller
constructor = Constructor(
    arena_conf=arena_config,
    mission_conf=missionConfig,
    objective_conf=objectiveConfig,
    ctrl_conf=specific_settings,
    observer_conf={"observer": None},
    timeout_in_sec=simulation_timeout,
    throw_exceptions=False,
)
# Step 1.1 Retrieve problem
problem = constructor.problem

# Step 1.2: Retrieve arena
arena = constructor.arena
observation = arena.reset(platform_state=problem.start_state)
problem_status = arena.problem_status(problem=problem)

# Step 2: Retrieve Controller
controller = constructor.controller
#%%
from ocean_navigation_simulator.utils.calc_fmrc_error import calc_fmrc_errors
calc_fmrc_errors(problem, arena, t_horizon_in_h=24, deg_around_x0_xT_box=0.5, T_goal_in_seconds=3600*24*2)
#%%
def calc_fmrc_errors(
    problem: Problem,
    arena: Arena,
    t_horizon_in_h: int,
    deg_around_x0_xT_box: float,
    T_goal_in_seconds: int,
)
# %%
deg_around_x0_xT_box = 0.5
T_horizon = 120
hours_to_abs_time = 3600

from ocean_navigation_simulator.utils.calc_fmrc_error import calc_fmrc_errors

calc_fmrc_errors(problem, T_horizon, deg_around_x0_xT_box, hours_to_abs_time=hours_to_abs_time)
# %%
# Step 1: extract data from them
(
    t_interval_global,
    lat_interval,
    lon_interval,
) = utils.simulation_utils.convert_to_lat_lon_time_bounds(
    problem.x_0,
    problem.x_T,
    deg_around_x0_xT_box=deg_around_x0_xT_box,
    temp_horizon_in_h=T_horizon,
    hours_to_hj_solve_timescale=hours_to_abs_time,
)

from datetime import datetime

# Definition of the forecast error here.
# Select the Forecast as basis
import xarray as xr

idx = 0
print(problem.forecast_data_source["content"][idx]["t_range"][0])
HYCOM_Forecast = xr.open_dataset(problem.forecast_data_source["content"][idx]["file"])
HYCOM_Forecast = HYCOM_Forecast.fillna(0).isel(depth=0)
# subset in time and space for mission

HYCOM_Forecast = HYCOM_Forecast.sel(
    time=slice(
        HYCOM_Forecast["time"].data[0],
        HYCOM_Forecast["time"].data[0] + np.timedelta64(T_horizon, "h"),
    ),
    lat=slice(lat_interval[0], lat_interval[1]),
    lon=slice(lon_interval[0], lon_interval[1]),
)
# %%
HYCOM_Forecast
# %%
t_interval = [HYCOM_Forecast.variables["time"][0], HYCOM_Forecast.variables["time"][-1]]
# %%
from datetime import timedelta

# t_interval_naive = [time.replace(tzinfo=None) for time in t_interval]
t_interval[0] = t_interval[0] - np.timedelta64(3, "h")
t_interval[1] = t_interval[1] + np.timedelta64(3, "h")
subsetted_frame = problem.hindcast_data_source["content"].sel(
    time=slice(t_interval[0], t_interval[1]),
    latitude=slice(lat_interval[0], lat_interval[1]),
    longitude=slice(lon_interval[0], lon_interval[1]),
)

DS_renamed_subsetted_frame = subsetted_frame.rename(
    {"vo": "water_v", "uo": "water_u", "latitude": "lat", "longitude": "lon"}
)
DS_renamed_subsetted_frame = DS_renamed_subsetted_frame.fillna(0)
Copernicus_right_time = DS_renamed_subsetted_frame.interp(time=HYCOM_Forecast["time"])
# interpolate 2D in space
Copernicus_H_final = Copernicus_right_time.interp(
    lon=HYCOM_Forecast["lon"].data, lat=HYCOM_Forecast["lat"].data, method="linear"
)

# again fill na with 0
Copernicus_H_final = Copernicus_H_final.fillna(0)

# %%

# %%
# Interpolate Hindcast to the Forecast axis
HYCOM_Hindcast = xr.open_mfdataset(
    [h_dict["file"] for h_dict in problem.hindcast_data_source["content"]]
)
HYCOM_Hindcast = HYCOM_Hindcast.fillna(0).isel(depth=0)
HYCOM_Hindcast["time"] = HYCOM_Hindcast["time"].dt.round("H")
HYCOM_Hindcast = HYCOM_Hindcast.sel(
    time=slice(t_interval[0], t_interval[1]),
    lat=slice(lat_interval[0], lat_interval[1]),
    lon=slice(lon_interval[0], lon_interval[1]),
)
# %%
u_data_forecast = HYCOM_Forecast["water_u"].data
v_data_forecast = HYCOM_Forecast["water_v"].data
u_data_hindcast = HYCOM_Hindcast["water_u"].data
v_data_hindcast = HYCOM_Hindcast["water_v"].data
# %% calculate RMSE over time
HYCOM_Forecast_Error = HYCOM_Hindcast - HYCOM_Forecast
water_u_error = HYCOM_Forecast_Error["water_u"].data
water_v_error = HYCOM_Forecast_Error["water_v"].data
magnitude_over_time = ((water_u_error) ** 2 + (water_v_error) ** 2) ** 0.5
RMSE = magnitude_over_time.mean(axis=(1, 2)).compute()
# %% calculate angle difference over time
np.abs(
    np.arctan2(v_data_hindcast, u_data_hindcast) - np.arctan2(v_data_forecast, u_data_forecast)
).mean(axis=[1, 2])
# %%
from ocean_navigation_simulator.utils.calc_fmrc_error import (
    calc_abs_angle_difference,
    calc_speed_RMSE,
    calc_vector_corr_over_time,
)

# Step 2: Calculate things and return them as dict
RMSE = calc_speed_RMSE(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast)
angle_diff = calc_abs_angle_difference(
    u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast
)
vec_corr = calc_vector_corr_over_time(
    HYCOM_Forecast.to_array().to_numpy().transpose((1, 2, 3, 0)),
    HYCOM_Hindcast.to_array().to_numpy().transpose((1, 2, 3, 0)),
)
# %%
dict_of_relevant_fmrcs = list(
    filter(
        lambda dic: dic["t_range"][0] < problem.t_0 + timedelta(hours=T_horizon),
        problem.forecast_data_source["content"],
    )
)
print(len(dics_containing_t_0))

# %%
RMSE_mean_across_fmrc = []
for i in range(1):
    RMSE_mean_across_fmrc.append(np.arange(10))

# %%
prob.viz(
    time=datetime.datetime(2021, 11, 26, 10, 10, 10, tzinfo=datetime.timezone.utc),
    cut_out_in_deg=1.2,
    plot_start_target=False,
)  # plots the current at t_0 with the start and goal position
# # create a video of the underlying currents rendered in Jupyter, Safari or as file
# prob.viz(video=True) # renders in Jupyter
# prob.viz(video=True, html_render='safari')    # renders in Safari
# prob.viz(video=True, filename='problem_animation.gif', temp_horizon_viz_in_h=200) # saves as gif file
# %% Check feasibility of the problem
feasible, earliest_T, gt_run_simulator = utils.check_feasibility_2D_w_sim(
    problem=prob,
    T_hours_forward=95,
    deg_around_xt_xT_box=1.2,
    sim_dt=7200,
    conv_m_to_deg=111120,
    grid_res=[0.04, 0.04],
    hours_to_hj_solve_timescale=3600,
    sim_deg_around_x_t=4,
)
# 67.0 exactly!
# %%
plt.rcParams["font.size"] = "40"
plt.rcParams["font.family"] = "Palatino"
# %
# ax = prob.viz(cut_out_in_deg=1.2, return_ax=True, alpha_currents=0.2, plot_start_target=False)
ax = gt_run_simulator.high_level_planner.plot_reachability_snapshot(
    rel_time_in_h=5,
    multi_reachability=True,
    granularity_in_h=10.0,
    alpha_color=1.0,
    time_to_reach=True,
    return_ax=True,
    fig_size_inches=(4 * 3.5, 4 * 2.5),
)
plt.show()

# %% Make the colorbar fit finally
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


im = plt.imshow(np.arange(200).reshape((20, 10)))
add_colorbar(im)

# %% plot 3D
from scipy.interpolate import interp1d

rel_time_in_h = 27
# get_initial_value
initial_values = gt_run_simulator.high_level_planner.get_initial_values(
    direction=gt_run_simulator.high_level_planner.specific_settings["direction"]
)

# interpolate
val_at_t = interp1d(
    gt_run_simulator.high_level_planner.reach_times
    - gt_run_simulator.high_level_planner.reach_times[0],
    gt_run_simulator.high_level_planner.all_values,
    axis=0,
    kind="linear",
)(
    rel_time_in_h
    * gt_run_simulator.high_level_planner.specific_settings["hours_to_hj_solve_timescale"]
).squeeze()

val_below_0 = np.where(val_at_t > 0, np.NaN, val_at_t)
# hj.viz.visFunc(gt_run_simulator.high_level_planner.grid,
#                val_at_t, color='black', alpha=0.5, marker=None)

grid = gt_run_simulator.high_level_planner.grid
data = val_below_0
color = "black"
alpha = 0.5
# %%
import plotly.io as pio

pio.renderers.default = "browser"
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Surface(
            opacity=alpha, x=grid.states[:, :, 0], y=grid.states[:, :, 1], z=data, connectgaps=False
        )
    ]
)
fig.update_traces(
    contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
)

fig.show()

# %%
gt_run_simulator.high_level_planner.plot_reachability_animation(
    type="mp4", multi_reachability=True, granularity_in_h=5, time_to_reach=True
)
# %% Plot the time-optimal trajectory (if true currents are known)
# gt_run_simulator.plot_trajectory(plotting_type='2D')
# gt_run_simulator.plot_trajectory(plotting_type='ctrl')
gt_run_simulator.plot_trajectory(plotting_type="2D_w_currents_w_controls")
# %%
gt_run_simulator.create_animation(
    vid_file_name="simulation_animation.gif",
    deg_around_x0_xT_box=1.2,
    temporal_stride=6,
    time_interval_between_pics=200,
    linewidth=4,
    marker=None,
    linestyle="-",
    add_ax_func=None,
)
# %% Set-Up and run Simulator
sim = OceanNavSimulator(
    sim_config_dict="simulator.yaml", control_config_dict="reach_controller.yaml", problem=prob
)
start = time.time()
sim.evaluation_run(T_in_h=10)
print("Took : ", time.time() - start)
print("Arrived after {} h".format((sim.trajectory[3, -1] - sim.trajectory[3, 0]) / 3600))
# 67.0 exactly!
# %% Step 5: plot results from Simulator
# # plot Battery levels over time
sim.plot_trajectory(plotting_type="battery")
# # # plot 2D Trajectory without background currents
sim.plot_trajectory(plotting_type="2D")
# # plot control over time
sim.plot_trajectory(plotting_type="ctrl")
# # plot 2D Trajectory with currents at t_0
sim.plot_trajectory(plotting_type="2D_w_currents")
# # plot 2D Trajectory with currents at t_0 and control_vec for each point
sim.plot_trajectory(plotting_type="2D_w_currents_w_controls")
# plot with plans
# sim.plot_2D_traj_w_plans(plan_idx=None, with_control=False, underlying_plot_type='2D', plan_temporal_stride=1)
# %% Step 5:2 create animations of the simulation run
sim.create_animation(
    vid_file_name="simulation_animation.mp4",
    deg_around_x0_xT_box=0.5,
    temporal_stride=1,
    time_interval_between_pics=200,
    linewidth=1.5,
    marker="x",
    linestyle="--",
    add_ax_func=None,
)
# %% Test how I can add the best-in-hindsight trajectory to it
from functools import partial

x_best = np.linspace(x_0[0], x_T[0], 20).reshape(1, -1)
y_best = np.linspace(x_0[1], x_T[1], 20).reshape(1, -1)
x_traj_best = np.vstack((x_best, y_best))


def plot_best(ax, time, x_traj_best):
    ax.plot(x_traj_best[0, :], x_traj_best[1, :], color="k", label="best in hindsight")


to_add = partial(plot_best, x_traj_best=x_traj_best)
sim.create_animation_with_plans(
    vid_file_name="simulation_animation_with_plans.mp4",
    plan_temporal_stride=5,
    alpha=1,
    with_control=True,
    plan_traj_color="sienna",
    plan_u_color="blue",
    add_ax_func=to_add,
)
# %% Plot the latest Plan from reachability planner
sim.high_level_planner.plot_2D_traj()
sim.high_level_planner.plot_ctrl_seq()
# %%
gt_run_simulator.high_level_planner.plot_reachability_snapshot(
    rel_time_in_h=0, multi_reachability=True, granularity_in_h=5.0, time_to_reach=True
)
# %%
import matplotlib.pyplot as plt

self = gt_run_simulator.high_level_planner
# gt_run_simulator.high_level_planner.vis_Value_func_along_traj()
figsize = (12, 12)
return_ax = False
extra_traj = None
time_to_reach = True
fig, ax = plt.subplots(figsize=figsize)

if time_to_reach:
    all_values_dimensional = (
        1 + self.all_values - (self.reach_times / self.reach_times[-1]).reshape(-1, 1, 1)
    )
    all_values = all_values_dimensional * self.specific_settings["T_goal_in_h"]
    ylabel = "Earliest-time-to-reach"
else:
    ylabel = r"$\phi(x_t)$"
    all_values = self.all_values

reach_times = (self.reach_times - self.reach_times[0]) / self.specific_settings[
    "hours_to_hj_solve_timescale"
]
traj_times = (
    self.planned_trajs[-1]["traj"][3, :] - self.current_data_t_0 - self.reach_times[0]
) / self.specific_settings["hours_to_hj_solve_timescale"]

hj.viz.visValFuncTraj(
    ax,
    traj_times=traj_times,
    x_traj=self.planned_trajs[-1]["traj"][:2, :],
    all_times=reach_times,
    all_values=all_values,
    grid=self.grid,
    flip_times=False,
)
# ylabel=ylabel)
if extra_traj is not None:
    extra_traj_times = (
        extra_traj[3, :] - self.current_data_t_0 - self.reach_times[0]
    ) / self.specific_settings["hours_to_hj_solve_timescale"]
    hj.viz.visValFuncTraj(
        ax,
        traj_times=extra_traj_times,
        x_traj=extra_traj[:2, :],
        all_times=self.reach_times / self.specific_settings["hours_to_hj_solve_timescale"],
        all_values=all_values,
        grid=self.grid,
        flip_times=False,
    )
    # ylabel=ylabel)
# %% Plot 2D reachable set evolution

# %%
sim.high_level_planner.plot_reachability_animation(
    type="mp4", multi_reachability=True, granularity_in_h=1, time_to_reach=True
)
