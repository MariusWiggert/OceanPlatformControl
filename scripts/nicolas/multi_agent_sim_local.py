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
import numpy as np

## Only when developing with VSCode in my repo
# os.chdir("/home/nicolas/documents/Master_Thesis_repo/OceanPlatformControl")
# print(os.getcwd())
##

logging.basicConfig(level=logging.INFO)

# Settings for where the problem is saved
my_path = os.getcwd()
save_in_folder = os.path.join(my_path, "generated_media/HC_HC/4_platforms/mission_285")
os.makedirs(save_in_folder, exist_ok=True)

NoObserver = {"observer": None}

# Controller Configs
HJMultiTimeConfig = {
    "replan_every_X_seconds": None,
    "replan_on_new_fmrc": True,
    "T_goal_in_seconds": 3600 * 24 * 5,  # 3d, 43200,     # 12h
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
    "optim_horizon": 6,  # 1h
    "scaling_pot_function": 5,
}
MultiAgentCtrlConfig = {
    "ctrl_name": "ocean_navigation_simulator.controllers.MultiAgentPlanner.MultiAgentPlanner",
    "high_level_ctrl": "multi_ag_optimizer",  # choose from hj_naive, flocking, reactive_control, multi_ag_optimizer
    "unit": "km",
    "communication_thrsld": 9,
    "hj_specific_settings": HJMultiTimeConfig,
    "flocking_config": flockingConfig,
    "reactive_control_config": reactiveConfig,
    "multi_ag_optim": multiAgentOptimConfig,
}
# Task Configs
# missionConfig = {
#     "feasible": True,
#     "seed": 571402,
#     "target_radius": 0.1,
#     "ttr_in_h": 60,  # here does not really make sense as it is normally computed by the missionGenerator
#     "x_0": [
#         {
#             "date_time": "2021-11-24T12:00:48+00:00",
#             "lat": 23.2,
#             "lon": -83.2,
#         },
#         {
#             "date_time": "2021-11-24T12:00:48+00:00",
#             "lat": 23.25,
#             "lon": -83.25,
#         },
#         {
#             "date_time": "2021-11-24T12:00:48+00:00",
#             "lat": 23.3,
#             "lon": -83.3,
#         },
#         {
#             "date_time": "2021-11-24T12:00:48+00:00",
#             "lat": 23.35,
#             "lon": -83.35,
#         },
#     ],
#     "x_T": {"lat": 24.35, "lon": -82.3},
# }

# mission failing for flocking nr 285
missionConfig = {
    "feasible": True,
    "seed": 571402,
    "target_radius": 0.1,
    "ttr_in_h": 60,  # here does not really make sense as it is normally computed by the missionGenerator
    "x_0": [
        {
            "date_time": "2022-05-16T12:49:54+00:00",
            "lat": 25.69220733642578,
            "lon": -87.30220794677734,
        },
        {
            "date_time": "2022-05-16T12:49:54+00:00",
            "lat": 25.64868927001953,
            "lon": -87.34028625488281,
        },
        {
            "date_time": "2022-05-16T12:49:54+00:00",
            "lat": 25.60809898376465,
            "lon": -87.39879608154297,
        },
        {
            "date_time": "2022-05-16T12:49:54+00:00",
            "lat": 25.57320785522461,
            "lon": -87.39673614501953,
        },
    ],
    "x_T": {"lat": 27.03541637, "lon": -86.93996833},
}

# missionConfig = {
#     "feasible": True,
#     "seed": 571402,
#     "target_radius": 0.1,
#     "ttr_in_h": 60,  # here does not really make sense as it is normally computed by the missionGenerator
#     "x_0": [
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.12441062927246,
#             "lon": -88.06990814208984,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.154739379882812,
#             "lon": -88.0869369506836,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.101789474487305,
#             "lon": -88.0881118774414,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.100204467773438,
#             "lon": -88.09916687011719,
#         },
#     ],
#     "x_T": {"lat": 22.31922451, "lon": -88.90433564},
# }
# missionConfig = {
#     "feasible": True,
#     "seed": 571402,
#     "target_radius": 0.1,
#     "ttr_in_h": 60,  # here does not really make sense as it is normally computed by the missionGenerator
#     "x_0": [
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.12441062927246,
#             "lon": -88.06990814208984,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.154739379882812,
#             "lon": -88.0869369506836,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.101789474487305,
#             "lon": -88.0881118774414,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.100204467773438,
#             "lon": -88.09916687011719,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.180204467773438,
#             "lon": -88.1016687011719,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.160204467773438,
#             "lon": -88.0616687011719,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.190204467773438,
#             "lon": -88.0916687011719,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.220204467773438,
#             "lon": -88.0816687011719,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.220204467773438,
#             "lon": -88.0516687011719,
#         },
#         {
#             "date_time": "2022-08-31T02:11:19+00:00",
#             "lat": 22.200204467773438,
#             "lon": -88.0316687011719,
#         },
#     ],
#     "x_T": {"lat": 22.31922451, "lon": -88.90433564},
# }


objective_conf = {"type": "nav"}
arenaConfig = {
    "casadi_cache_dict": {"deg_around_x_t": 0.5, "time_around_x_t": 86400.0},
    "ocean_dict": {
        "hindcast": {
            "field": "OceanCurrents",
            "source": "hindcast_files",
            "source_settings": {
                "folder": "data/miss_gen_hindcast/",
                "local": False,
                "source": "HYCOM",
                "type": "hindcast",
                "currents": "total",
                "region": "GOM",
                # "region": "Region 1",
            },
        },
        "forecast": None,  # {
        #     "field": "OceanCurrents",
        #     "source": "forecast_files",
        #     "source_settings": {
        #         "folder": "data/miss_gen_forecast/",
        #         "local": False,
        #         "source": "Copernicus",
        #         "type": "forecast",
        #         "currents": "total",
        #         "region": "GOM",
        #     },
        # },
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
    "timeout": 3600 * 24 * 4,
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
    download_files=True,  # True,,
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
#     x_0=problem.start_state.to_spatio_temporal_point(),
#     x_T=problem.end_region,
#     deg_around_x0_xT_box=0.24,
#     temp_horizon_in_s=3600,
# )

# ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=problem.start_state.date_time[0],
#     x_interval=lon_bnds,
#     y_interval=lat_bnds,
#     return_ax=True,
#     vmax=0.45,
# )
# problem.plot(ax=ax)
# plt.show()

#%%
# Step 2: Retrieve Controller
# problem.platform_dict = arena.platform.platform_dict
controller = constructor.controller
# Reachability snapshot plot
action = controller.get_action(observation=observation)

# %%
# Step 3: Retrieve observer
observer = constructor.observer
# Step 4: Run closed-loop simulation
ctrl_deviation_from_opt = []
all_pltf_status = [0] * len(missionConfig["x_0"])
min_distances_to_target_over_mission = [np.inf] * len(missionConfig["x_0"])
pb_running_thrsld = 0
# Run until tiemout of until one of the platform has stranded/left arena region (failed)
while not any(status < pb_running_thrsld for status in problem_status):
    # Get action
    action, ctrl_correction = controller.get_action(
        observation=observation
    )  # correction angle in rad
    ctrl_deviation_from_opt.append(ctrl_correction)
    # execute action
    observation = arena.step(action)

    # Observer data assimilation
    # observer.observe(observation)
    # observation.forecast_data_source = observer
    # #this replaces the forecast source by Observer (defined as none for now so not a desired behavior)

    # update problem status
    problem_status = arena.problem_status(problem=problem)
    min_distances_to_target_over_mission = arena.get_min_or_max_of_two_lists(
        list_a=min_distances_to_target_over_mission,
        list_b=arena.final_distance_to_target(problem=problem),
        min_or_max="min",
    )
    # for the final metric, look if platform was able to reach target within T, so keep only max (=1 if pltf reached target)
    all_pltf_status = arena.get_min_or_max_of_two_lists(
        list_a=all_pltf_status, list_b=problem_status, min_or_max="max"
    )
print("terminated because:", arena.problem_status_text(arena.problem_status(problem=problem)))
controller.plot_reachability_snapshot_over_currents(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    return_ax=True,
)
plt.show()

# %% Plot useful metrics for multi-agent performance evaluation
results_folder = os.path.join(
    save_in_folder, MultiAgentCtrlConfig["high_level_ctrl"] + "_full_horiz"
)
os.makedirs(results_folder, exist_ok=True)

with open(f"{results_folder}/missionConfig.yml", "w") as outfile:
    yaml.dump(missionConfig, outfile, default_flow_style=False)

metrics_dict = arena.save_metrics_to_log(
    all_pltf_status=all_pltf_status,
    min_distances_to_target=min_distances_to_target_over_mission,
    max_correction_from_opt_ctrl=ctrl_deviation_from_opt,
    filename=f"{results_folder}/metrics.log",
)
metrics_df = pd.DataFrame(data=metrics_dict, index=[0])
metrics_df.to_csv(f"{results_folder}/metrics.csv")
arena.animate_trajectory(
    margin=0.1,
    problem=problem,
    temporal_resolution=7200,
    output=f"{results_folder}/trajectory_anim.mp4",
    fps=6,
    ctrl_scale=20,
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
arena.plot_all_on_map(problem=problem, show_control_trajectory=False, margin=0.25, return_ax=True)
plt.savefig(f"{results_folder}/state_trajectory.png")

plt.clf()
arena.plot_distance_evolution_between_platforms()
plt.savefig(f"{results_folder}/distanceEvolution.png")
# %%
