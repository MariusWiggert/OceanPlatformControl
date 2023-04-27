# This script shows how the closed-loop simulation is run in the C3 cloud, using only configs and the constructor
# Note: This Mission is successful with Naive but not with HJ closed-loop planning.
# %% imports
import logging

from ocean_navigation_simulator.problem_factories.Constructor import (
    Constructor,
)

logging.basicConfig(level=logging.INFO)
# % configs

# Observer Configs
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

# Task Configs
missionConfig = {
    "feasible": True,
    "seed": 571402,
    "target_radius": 0.1,
    "ttr_in_h": 90.63763978820165,
    "x_0": [
        {
            "date_time": "2022-09-26T21:29:48+00:00",
            "lat": 27.522615432739258,
            "lon": -85.29534912109375,
        }
    ],
    "x_T": {"lat": 27.02574792857954, "lon": -86.11192123175897},
}
objective_conf = {"type": "nav"}
arenaConfig = {
    "casadi_cache_dict": {"deg_around_x_t": 0.5, "time_around_x_t": 86400.0},
    "ocean_dict": {
        "region": "GOM", # this is gulf of mexico
        "forecast": {
            "field": "OceanCurrents",
            "source": "forecast_files",
            "source_settings": {
                "folder": "tmp/forecast/",
                "source": "Copernicus",
                "type": "forecast",
            },
        },
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
    "timeout": 3600 * 24 * 5, # timeout of simulation
}

# Step 0: Create Constructor object which contains arena, problem, controller and observer
constructor = Constructor(
    arena_conf=arenaConfig,
    mission_conf=missionConfig,
    objective_conf=objective_conf,
    ctrl_conf=StraightLineConfig,  # here different controller configs can be put in
    observer_conf=NoObserver,  # here the other observers can also be put int
    download_files=True,
    timeout_in_sec=arenaConfig["timeout"],
)

# Step 1.1 Retrieve problem
problem = constructor.problem

# Step 1.2: Retrieve arena
arena = constructor.arena
observation = arena.reset(platform_state=problem.start_state)
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
# # If you want to visualize the plan beforehand (only with Multi-Time HJ Controller)
# action = controller.get_action(observation=observation)
# controller.plot_reachability_snapshot(
#     rel_time_in_seconds=0,
#     granularity_in_h=2,
#     alpha_color=1,
#     time_to_reach=True,
#     fig_size_inches=(12, 12),
#     plot_in_h=True,
# )

#%
# Step 3: Retrieve observer
observer = constructor.observer

# Step 4: Run closed-loop simulation
while problem_status == 0:
    # Get action
    action = controller.get_action(observation=observation)

    # execute action
    observation = arena.step(action)

    # Observer data assimilation
    observer.observe(observation)
    observation.forecast_data_source = observer

    # update problem status
    problem_status = arena.problem_status(problem=problem)

print("terminated because:", arena.problem_status_text(arena.problem_status(problem=problem)))
#%% Additional plotting and animation lines
arena.plot_all_on_map(problem=problem)
#%%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=7200,
    output="evaluation_runner_animation.mp4",
)
