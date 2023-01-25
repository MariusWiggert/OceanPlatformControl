# This script shows how the closed-loop simulation is run in the C3 cloud, using only configs and the constructor
# %% imports
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from ocean_navigation_simulator.problem_factories.Constructor import (
    Constructor,
)
import yaml

## Only when developing with VSCode in my repo
os.chdir("/home/nicolas/documents/Master_Thesis_repo/OceanPlatformControl")
print(os.getcwd())
##

logging.basicConfig(level=logging.INFO)

# Settings for where the problem is saved
my_path = os.getcwd()
save_in_folder = os.path.join(my_path, "generated_media")
os.makedirs(save_in_folder, exist_ok=True)

NoObserver = {"observer": None}

# Controller Configs
HJMultiTimeConfig = {
    "replan_every_X_seconds": None,
    "replan_on_new_fmrc": True,
    "T_goal_in_seconds": 259200,  # 3d, 43200,     # 12h
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "ctrl_name": "ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner",
    "d_max": 0.0,
    "deg_around_xt_xT_box": 1.0,
    "direction": "multi-time-reach-back",
    "grid_res": 0.02,
    "n_time_vector": 200,
    "progress_bar": True,
    "use_geographic_coordinate_system": True,
}
StraightLineConfig = {
    "ctrl_name": "ocean_navigation_simulator.controllers.NaiveController.NaiveController"
}
flockingConfig = {
    "unit": "km",
    "interaction_range": 9,  # km
    "grad_clip_range": 0.1,  # km
}
reactiveConfig = {
    "unit": "m",
    "mix_ttr_and_euclidean": True,
    "delta_3": 8800,  # collision threshold (communication - delta_3)
    "delta_2": 2000,  # safe zone threshold  ]communication - delta_2, communication - delta_3[
    "delta_1": 500,  # small threshold so that if distance > communication_thrsld- delta_1 we try to achieve connectivity
    "communication_thrsld": 9000,
    "k_1": 0.25,
    "k_2": 1,
}
multiAgentOptimConfig = {
    "unit": "m",
    "interaction_range": 9000,  # m
    "optim_horizon": 2,
}
MultiAgentCtrlConfig = {
    "ctrl_name": "ocean_navigation_simulator.controllers.MultiAgentPlanner.MultiAgentPlanner",
    "high_level_ctrl": "multi_ag_optimizer",  # choose from hj_naive, flocking, reactive_control
    "unit": "km",
    "communication_thrsld": 9,
    "hj_specific_settings": HJMultiTimeConfig,
    "flocking_config": flockingConfig,
    "reactive_control_config": reactiveConfig,
    "multi_ag_optim": multiAgentOptimConfig,
}
# Task Configs
missionConfig = {
    "feasible": True,
    "seed": 571402,
    "target_radius": 0.1,
    "ttr_in_h": 90.63763978820165,
    "x_0": [
        {
            "date_time": "2021-11-24T12:00:48+00:00",
            "lat": 23.2,
            "lon": -83.2,
        },
        {
            "date_time": "2021-11-24T12:00:48+00:00",
            "lat": 23.25,
            "lon": -83.25,
        },
        {
            "date_time": "2021-11-24T12:00:48+00:00",
            "lat": 23.3,
            "lon": -83.3,
        },
        {
            "date_time": "2021-11-24T12:00:48+00:00",
            "lat": 23.35,
            "lon": -83.35,
        },
    ],
    "x_T": {"lat": 24.35, "lon": -82.3},
}
objective_conf = {"type": "nav"}
arenaConfig = {
    "casadi_cache_dict": {"deg_around_x_t": 0.5, "time_around_x_t": 86400.0},
    "ocean_dict": {
        "area": "GOM",
        "forecast": {
            "field": "OceanCurrents",
            "source": "hindcast_files",
            "source_settings": {"folder": "tmp/hindcast/", "source": "HYCOM", "type": "hindcast"},
        },
        # {
        #     "field": "OceanCurrents",
        #     "source": "forecast_files",
        #     "source_settings": {
        #         "folder": "tmp/forecast/",
        #         "source": "Copernicus",
        #         "type": "forecast",
        #     },
        # },
        "hindcast": {
            "field": "OceanCurrents",
            "source": "hindcast_files",
            "source_settings": {"folder": "tmp/hindcast/", "source": "HYCOM", "type": "hindcast"},
        },
    },
    "platform_dict": {
        "battery_cap_in_wh": 400.0,
        "drag_factor": 675.0,
        "dt_in_s": 600.0,
        "motor_efficiency": 1.0,
        "solar_efficiency": 0.2,
        "solar_panel_size": 0.5,
        "u_max_in_mps": 0.1,
    },
    "seaweed_dict": {"forecast": None, "hindcast": None},
    "solar_dict": {"forecast": None, "hindcast": None},
    "spatial_boundary": None,
    "use_geographic_coordinate_system": True,
    "timeout": 3600 * 24 * 3,
    "multi_agent_constraints": {
        "unit": "km",
        "communication_thrsld": 9,
        "epsilon_margin": 1,  # when add edges based on hysteresis
        "collision_thrsld": 0.2,
    },
}

# Step 0: Create Constructor object which contains arena, problem, controller and observer
constructor = Constructor(
    arena_conf=arenaConfig,
    mission_conf=missionConfig,
    objective_conf=objective_conf,
    ctrl_conf=MultiAgentCtrlConfig,  # here different controller configs can be put in
    observer_conf=NoObserver,  # here the other observers can also be put int
    download_files=False,  # True,
    timeout_in_sec=arenaConfig["timeout"],
)
# Step 1.1 Retrieve problem
problem = constructor.problem

# Step 1.2: Retrieve arena
arena = constructor.arena
observation = arena.reset(platform_set=problem.start_state)
problem_status = arena.problem_status(problem=problem)
# #%% Plot the problem on the map
# import matplotlib.pyplot as plt
# t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
#     x_0=problem.start_state.to_spatio_temporal_point(), x_T=problem.end_region,
#     deg_around_x0_xT_box=1, temp_horizon_in_s=3600
# )
#
# ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=problem.start_state.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
# )
# problem.plot(ax=ax)
# plt.show()

#%%
# Step 2: Retrieve Controller
# problem.platform_dict = arena.platform.platform_dict
controller = constructor.controller
# Reachability snapshot plot
action = controller.get_action(observation=observation)
controller.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=True,
    return_ax=True,
)
plt.show()

# %%
# Step 3: Retrieve observer
observer = constructor.observer
# Step 4: Run closed-loop simulation
ctrl_deviation_from_opt = []
all_pltf_status = [0] * len(missionConfig["x_0"])
while any(status == 0 for status in problem_status):
    # Get action
    action, ctrl_correction = controller.get_action(observation=observation)
    ctrl_deviation_from_opt.append(ctrl_correction)
    # execute action
    observation = arena.step(action)

    # Observer data assimilation
    # observer.observe(observation)
    # observation.forecast_data_source = observer
    # #this replaces the forecast source by Observer (defined as none for now so not a desired behavior)

    # update problem status
    problem_status = arena.problem_status(problem=problem)

    # for the final metric, look if platform was able to reach target within T, so keep only max (=1 if pltf reached target)
    all_pltf_status = list(map(max, zip(all_pltf_status, problem_status)))

print("terminated because:", arena.problem_status_text(arena.problem_status(problem=problem)))

# %% Plot useful metrics for multi-agent performance evaluation
results_folder = os.path.join(save_in_folder, MultiAgentCtrlConfig["high_level_ctrl"])
os.makedirs(results_folder, exist_ok=True)

with open(f"{results_folder}/missionConfig.yml", "w") as outfile:
    yaml.dump(missionConfig, outfile, default_flow_style=False)

metrics_dict = arena.save_metrics_to_log(
    all_pltf_status=all_pltf_status,
    max_correction_from_opt_ctrl=ctrl_deviation_from_opt,
    filename=f"{results_folder}/metrics.log",
)
metrics_df = pd.DataFrame(data=metrics_dict, index=[0])
metrics_df.to_csv(f"{results_folder}/metrics.csv")
arena.animate_trajectory(
    margin=0.25,
    problem=problem,
    temporal_resolution=7200,
    output=f"{results_folder}/trajectory_anim.mp4",
    fps=6,
)

arena.animate_graph_net_trajectory(
    temporal_resolution=7200,
    # collision_communication_thrslds=(10, 50), (not specified take defaut one)
    plot_ax_ticks=True,
    output=f"{results_folder}/network_graph_anim.mp4",
    fps=5,
)

plt.clf()
fig = arena.plot_all_network_analysis(xticks_temporal_res=8 * 3600)  # 8 hours interval for xticks
plt.savefig(f"{results_folder}/graph_properties.png")
plt.clf()
arena.plot_distance_evolution_between_platforms()
plt.savefig(f"{results_folder}/distanceEvolution.png")
# %%
