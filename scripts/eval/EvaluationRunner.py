# %% imports
import datetime
import time

import yaml
import logging

# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

from ocean_navigation_simulator.problem_factories.Constructor import Constructor
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

# % configs
observer_config = {
    "observer": {
        "life_span_observations_in_sec": 86400,  # 24 * 3600
        "model": {
            "gaussian_process": {
                "sigma_noise_squared": 0.000005,
                # 3.6 ** 2 = 12.96
                "sigma_exp_squared": 100,  # 12.96
                "kernel": {
                    "scaling": {"latitude": 1, "longitude": 1, "time": 10000},  # [m]  # [m]  # [s]
                    "type": "matern",
                    "parameters": {"length_scale_bounds": "fixed"},
                },
                "time_horizon_predictions_in_sec": 3600,
            }
        },
    }
}

# x_0 = PlatformState(
#     lon=units.Distance(deg=-82.5),
#     lat=units.Distance(deg=23.7),
#     date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )

x_0 = {
    "lon": -82.5,
    "lat": 23.7,
    "date_time": "2022-08-24 23:12:00.004573 +0000",
}

# x_T = SpatialPoint(lon=units.Distance(deg=-80.3), lat=units.Distance(deg=24.6))

x_T = {"lon": -80.3, "lat": 24.6}

with open(f"config/arena/gulf_of_mexico_HYCOM_hindcast_local.yaml") as f:
    arena_config = yaml.load(f, Loader=yaml.FullLoader)

mission_config = {
    "x_0": [x_0],
    "x_T": x_T,
    "target_radius": 0.1,
    "seed": 12344093,
}
#%
ctrl_config = {
    "ctrl_name": "ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner",
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 4,
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
}

objective_conf = {"type": "nav"}

# #%%
# ArenaFactory.download_files(
#     config=arena_config, type="hindcast",
#     t_interval=[datetime.datetime(2021, 11, 25), datetime.datetime(2021, 11, 26)],
#     points=None, verbose=1)
# #%%
# ArenaFactory.download_required_files(
#         archive_source=arena_config["ocean_dict"]["hindcast"]["source_settings"][
#             "source"
#         ],
#         archive_type=arena_config["ocean_dict"]['hindcast']["source_settings"]["type"],
#         download_folder="data/hindcast_test_new/",
#         t_interval=[datetime.datetime(2021, 11, 25), datetime.datetime(2021, 11, 26)],
#         region=arena_config["ocean_dict"]["area"],
#         throw_exceptions=True,
#         points=None,
#         verbose=1
#     )
#%


## % run eval
mission_start_time = time.time()

# Step 0: Create Constructor object which contains arena, problem, controller and observer
constructor = Constructor(
    arena_conf=arena_config,
    mission_conf=mission_config,
    objective_conf=objective_conf,
    ctrl_conf=ctrl_config,
    observer_conf=observer_config,
)


# Step 1.1 Retrieve problem
problem = constructor.problem

# Step 1.2: Retrieve arena
arena = constructor.arena
observation = arena.reset(platform_state=problem.start_state)
problem_status = arena.problem_status(problem=problem)
# % Plot the problem on the map
# t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
#     x_0=x_0.to_spatio_temporal_point(), x_T=x_T, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
# )

# ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
# )
# problem.plot(ax=ax)
# plt.show()

# Step 2: Retrieve Controller
# problem.platform_dict = arena.platform.platform_dict

controller = constructor.controller

action = controller.get_action(observation=observation)
controller.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=False,
    fig_size_inches=(12, 12),
    plot_in_h=True,
)

# Step 3: Retrieve observer
observer = constructor.observer


# Step 4: Run Arena
# TODO: investigate runtime of collision check
while problem_status == 0:
    # Observer data assimilation
    observer.observe(observation)
    observation.forecast_data_source = observer
    # Get action
    action = controller.get_action(observation=observation)

    # execute action
    observation = arena.step(action)

    # update problem status
    problem_status = arena.problem_status(problem=problem)
#%%
arena.plot_all_on_map(problem=problem)
#%%
arena.animate_trajectory(problem=problem, temporal_resolution=7200)
# %%
