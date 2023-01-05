#%%
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

os.chdir("/home/nicolas/documents/Master_Thesis_repo/OceanPlatformControl")
# os.chdir("/home/nicolas/codeRepo/OceanPlatformControl")
from ocean_navigation_simulator.controllers.Multi_agent_planner import (
    MultiAgentPlanner,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import (
    PlatformState,
    PlatformStateSet,
)
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units

#%%  Import scenario and configurate folder to save plots (for analysis)
multi_agent_scenario = "scenario_2"
with open(f"config/multi_agent_scenarios/{multi_agent_scenario}.yaml") as f:
    multi_ag_config = yaml.load(f, Loader=yaml.FullLoader)

my_path = os.path.abspath("/home/nicolas/documents/Master_Thesis_repo/OceanPlatformControl")
folder_save_results = os.path.join(my_path, f'generated_media/{multi_ag_config["folder_to_save"]}')
if not os.path.isdir(folder_save_results):
    os.makedirs(folder_save_results)
platf_init_dict = multi_ag_config["platforms_start"]
target_dict = multi_ag_config["target_region"]

# save yaml to result folder
with open(f"{folder_save_results}/config.yaml", "w") as fp:
    yaml.dump(multi_ag_config, fp)
#%% Initialize the Arena, target region and the platform states
arena = ArenaFactory.create(
    scenario_name=multi_ag_config["data_source_config"],
    scenario_config_multi_agent=multi_ag_config["multi_agent_param"],
)
x_T = SpatialPoint(
    lon=units.Distance(deg=target_dict["lon"]), lat=units.Distance(deg=target_dict["lat"])
)
x_s = []  # list of platform states starts
for idx in range(platf_init_dict["nb_platforms"]):
    x_s.append(
        PlatformState(
            lon=units.Distance(deg=platf_init_dict["lon"][idx]),
            lat=units.Distance(deg=platf_init_dict["lat"][idx]),
            date_time=datetime.datetime(
                platf_init_dict["year"],
                platf_init_dict["month"],
                platf_init_dict["day"],
                platf_init_dict["hours"],
                platf_init_dict["minutes"],
                tzinfo=datetime.timezone.utc,
            ),
        )
    )
platform_set = PlatformStateSet(x_s)
#%% create a navigation problem
problem = NavigationProblem(
    start_state=platform_set,
    end_region=x_T,
    target_radius=target_dict["radius"],
    timeout=datetime.timedelta(days=target_dict["timeout_days"]),
    platform_dict=arena.platform.platform_dict,
)

# %% Plot the problem on the map
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=platform_set.to_spatio_temporal_point(),
    x_T=x_T,
    deg_around_x0_xT_box=1,
    temp_horizon_in_s=3600,
)
ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=platform_set.get_date_time(),
    x_interval=lon_bnds,
    y_interval=lat_bnds,
    return_ax=True,
    figsize=(12, 12),
)
problem.plot(ax=ax)
plt.savefig(f"{folder_save_results}/InitialProblem.png")

# %% Instantiate the Multi-Agent using HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * multi_ag_config["days_sim"],
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "initial_set_radii": [
        0.1,
        0.1,
    ],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
    # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
    "platform_dict": arena.platform.platform_dict,
}
# multi_agent_planner_settings = {"planner": "hj_planner"
#                         "": {"communication_thrld": multi_ag_config["multi_agent_param"]

#                         }}
planner_set = MultiAgentPlanner(
    problem=problem,
    multi_agent_settings=multi_ag_config["multi_agent_param"],
    specific_settings=specific_settings,
)
# first observation of initial states
observation = arena.reset(platform_set=platform_set)
action = planner_set.get_action_HJ_naive(observation=observation)  # get first action to take
all_pltf_status = [
    arena.problem_status(problem=problem, platform_id=id) for id in range(len(platform_set))
]
# %% Reachability snapshot plot
plt.clf()
planner_set.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=True,
    return_ax=True,
)
plt.savefig(f"{folder_save_results}/ReachabilitySnap.png")

# %% Simulate a trajectory:
update_rate_s = arena.platform.platform_dict["dt_in_s"]  # 10 mins
day_sim = multi_ag_config["days_sim"]
max_flock_correction_list = []
for i in tqdm(range(int(3600 * 24 * day_sim / update_rate_s))):  #
    action, max_flock_correction = planner_set.get_action_HJ_with_flocking(observation=observation)
    observation = arena.step(action)
    # check if platforms have reached target
    new_status = [
        arena.problem_status(problem=problem, platform_id=id) for id in range(len(platform_set))
    ]
    # for the final metric, look if platform was able to reach target within T, so keep only max (=1 if pltf reached target)
    all_pltf_status = list(map(max, zip(all_pltf_status, new_status)))
    max_flock_correction_list.append(max_flock_correction)

metrics_dict = arena.save_metrics_to_log(
    all_pltf_status=all_pltf_status,
    max_correction_from_opt_ctrl=max_flock_correction_list,
    filename=f"{folder_save_results}/metrics.log",
)
metrics_df = pd.DataFrame(data=metrics_dict, index=[0])
metrics_df.to_csv(f"{folder_save_results}/metrics.csv")
# animations
arena.animate_trajectory(
    margin=0.25,
    problem=problem,
    temporal_resolution=7200,
    output=f"{folder_save_results}/trajectory_anim.mp4",
    fps=6,
)

arena.animate_graph_net_trajectory(
    temporal_resolution=7200,
    # collision_communication_thrslds=(10, 50), (not specified take defaut one)
    plot_ax_ticks=True,
    output=f"{folder_save_results}/network_graph_anim.mp4",
    fps=5,
)
# Plot trajectory
plt.clf()
arena.plot_all_on_map(problem=problem, return_ax=True, figsize=(10, 8))
plt.savefig(f"{folder_save_results}/trajectory_plot.png")
# %% Plot useful metrics for multi-agent performance evaluation
plt.clf()
fig = arena.plot_all_network_analysis(xticks_temporal_res=8 * 3600)  # 8 hours interval for xticks
plt.savefig(f"{folder_save_results}/graph_properties.png")
plt.clf()
arena.plot_distance_evolution_between_platforms()
plt.savefig(f"{folder_save_results}/distanceEvolution.png")
