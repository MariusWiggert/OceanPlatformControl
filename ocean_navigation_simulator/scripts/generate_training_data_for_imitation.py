import string
import time
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
import os
import gc

from ocean_navigation_simulator.env.Arena import ArenaObservation, Arena
from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.Platform import PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint
from ocean_navigation_simulator.env.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.controllers.HjPlanners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.problem import Problem

def generate_training_data_for_imitation(
    arena: Arena,
    problem: Problem,
    mission_folder: string,
    steps: int=600,
    verbose: int=1,
    plot_after: int=100,
):
    start = time.time()
    # Instantiate the HJ Planners
    specific_settings = {
        'replan_on_new_fmrc': True,
        'replan_every_X_seconds': None,
        'direction': 'multi-time-reach-back',
        'n_time_vector': 100,  # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
        'deg_around_xt_xT_box': 1.,  # area over which to run HJ_reachability
        'accuracy': 'high',
        'artificial_dissipation_scheme': 'local_local',
        'T_goal_in_seconds': problem.timeout,
        'use_geographic_coordinate_system': True,
        'progress_bar': True,
        'initial_set_radii': [0.1, 0.1],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
        # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
        'grid_res': 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
        'd_max': 0.0,
        # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
        # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
        'platform_dict': arena.platform.platform_dict
    }
    global planner_hycom
    planner_hycom = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
    planner_copernicus = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)

    # Feature Settings
    TRUE_CURRENT_LENGTH = 5
    TTR_MAP_IN_WIDTH = 15
    TTR_MAP_IN_WIDTH_DEG = 0.25
    TTR_MAP_OUT_WIDTH = 3
    TTR_MAP_OUT_WIDTH_DEG = 0.25/5

    trajectory = []

    x_mission = np.zeros((0, TRUE_CURRENT_LENGTH * 4 + TTR_MAP_IN_WIDTH * TTR_MAP_IN_WIDTH))
    y_mission = np.zeros((0, 3 * 3))

    observation = arena.reset(platform_state=problem.start_state)

    print(f'Planner created: ({time.time() - start:.1f}s)')

    for t in tqdm(range(steps), disable=verbose<1):
        start = time.time()
        action = planner_hycom.get_action(observation=observation)
        planner_copernicus.get_action(
            observation=ArenaObservation(
                platform_state = observation.platform_state,
                true_current_at_state = observation.true_current_at_state,
                forecast_data_source = arena.ocean_field.hindcast_data_source
            )
        )
        print(f'Planner get action: ({time.time() - start:.2f}s)')
        start = time.time()
        observation = arena.step(action)
        print(f'Arena step: ({time.time() - start:.2f}s)')
        start = time.time()

        point = {
            'datetime': observation.platform_state.date_time,
            'lon': observation.platform_state.lon.deg,
            'lat': observation.platform_state.lat.deg,
            'u_true': observation.true_current_at_state.u,
            'v_true': observation.true_current_at_state.v,
        }
        trajectory.append(point)

        def get_value_function_grid(planner, name, width, width_deg, save_to_csv=True):
            # Interpolate Temporal
            val_at_t = interp1d(planner.reach_times, planner.all_values, axis=0, kind='linear')(
                observation.platform_state.date_time.timestamp() - planner.current_data_t_0
            ).squeeze()
            val_at_t = (val_at_t - val_at_t.min()) * (planner.current_data_t_T - planner.current_data_t_0) / 3600

            # Interpolate Spacial
            in_grid_x = planner.grid.states[:, 0, 0]
            in_grid_y = planner.grid.states[0, :, 1]
            out_grid_x = np.linspace(point['lon'] - width_deg, point['lon'] + width_deg, width)
            out_grid_y = np.linspace(point['lat'] - width_deg, point['lat'] + width_deg, width)
            val_at_xy = interp2d(in_grid_y, in_grid_x, val_at_t, kind='linear')(
                out_grid_y, out_grid_x
            ).squeeze()

            # Debug Information
            if verbose > 0 and t > 0 and t % plot_after == 0:
                print(f'Time Scale: {(planner.current_data_t_T - planner.current_data_t_0) / 3600}h')
                print(f'Reach Times: [{planner.reach_times[0]}, {planner.reach_times[1]}, ..., {planner.reach_times[-1]}]')
                print(f'Relative Time:{observation.platform_state.date_time.timestamp() - planner.current_data_t_0}')

                min_index = np.unravel_index(np.argmin(val_at_t, axis=None), val_at_t.shape)
                max_index = np.unravel_index(np.argmax(val_at_t, axis=None), val_at_t.shape)
                print(f'Minimum: {val_at_t.min()} @ ({planner.grid.states[min_index]})')
                print(f'Minimum: {val_at_t.max()} @ ({planner.grid.states[max_index]})')

                CS = plt.contour(planner.grid.states[..., 0], planner.grid.states[..., 1], val_at_t, levels=np.arange(0,400,10))
                plt.clabel(CS, inline=True, fontsize=10)
                plt.scatter(x=point['lon'], y=point['lat'], c='r', marker='o')
                plt.scatter(planner.grid.states[..., 0].flatten(), planner.grid.states[..., 1].flatten(), s=0.05)
                plot_x, plot_y = np.meshgrid(in_grid_x, in_grid_y)
                plt.scatter(out_grid_x, plot_y, s=0.05, c='g')
                plot_x, plot_y = np.meshgrid(out_grid_x, out_grid_y)
                plt.scatter(out_grid_x, plot_y, s=0.05, c='r')
                plt.title(f'{name} ({mission_folder}): Iteration {t})')
                plt.show()

            return val_at_xy

        def get_x_y_train(val_hycom, val_copernicus, true_currents):
            true_currents[:, :2] = true_currents[:, :2] - np.ones((5, 2)) * true_currents[-1, :2]
            ttr_map_out = val_copernicus - val_hycom[7:10, 7:10]

            x = np.concatenate((true_currents.flatten(), val_hycom.flatten()))
            y = ttr_map_out.flatten()

            return x, y

        # Generate Training Data
        if t >= TRUE_CURRENT_LENGTH:
            val_hycom = get_value_function_grid(planner_hycom, 'HYCOM', TTR_MAP_IN_WIDTH, TTR_MAP_IN_WIDTH_DEG, save_to_csv=False)
            val_copernicus = get_value_function_grid(planner_copernicus, 'Copernicus', TTR_MAP_OUT_WIDTH, TTR_MAP_OUT_WIDTH_DEG, save_to_csv=False)
            true_currents = np.array([np.array([t['lon'], t['lat'], t['u_true'], t['v_true']]) for t in trajectory[-TRUE_CURRENT_LENGTH:]])

            x_train, y_train = get_x_y_train(val_hycom, val_copernicus, true_currents)

            x_mission = np.append(x_mission, np.expand_dims(x_train.squeeze(), axis=0), axis=0)
            y_mission = np.append(y_mission, np.expand_dims(y_train.squeeze(), axis=0), axis=0)

        print(f'Data generation: ({time.time() - start:.2f}s)')

        # Simulation Termination
        prolem_status = problem.is_done(observation.platform_state)
        if prolem_status == 1 or prolem_status == -1 or not arena.is_inside_arena():
            break

    # Save Data
    # if not os.path.exists(mission_folder):
    #     os.mkdir(mission_folder)
    # pd.DataFrame(trajectory).to_csv(f'{mission_folder}/trajectory.csv')
    # with open(f'{mission_folder}/x_mission.npy', 'wb') as f:
    #     np.save(f, x_mission)
    # with open(f'{mission_folder}/y_mission.npy', 'wb') as f:
    #     np.save(f, y_mission)

    # Delete Objects
    del planner_hycom
    del planner_copernicus
    del x_mission
    del y_mission
    del trajectory
    gc.collect()


if __name__ == "__main__":
    script_start_time = time.time()

    arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast')

    problem = NavigationProblem(
        start_state=PlatformState(
            lon=units.Distance(deg=-81.74966879179048),
            lat=units.Distance(deg=18.839454259572026),
            date_time=datetime.datetime(2021, 11, 24, 12, 10, tzinfo=datetime.timezone.utc)
        ),
        end_region=SpatialPoint(
            lon=units.Distance(deg=-83.17890714569597),
            lat=units.Distance(deg=18.946404547127734)
        ),
        target_radius=0.1,
        timeout=100*3600
    )

    generate_training_data_for_imitation(
        arena=arena,
        problem=problem,
        mission_folder='data/value_function_learning/test_mission',
        steps=10,
        verbose=1,
        plot_after=100,
    )

    print(f"Total Script Time: {time.time() - script_start_time:.2f}s = {(time.time() - script_start_time) / 60:.2f}min")


#%% Plots
# print(planner_hycom.reach_times)
# print(observation.platform_state.date_time.timestamp() - planner_hycom.current_data_t_0)

# x_interval, y_interval, t_interval =  arena.get_lon_lat_time_interval(end_region=problem.end_region, margin=1)
# ax = observation.forecast_data_source.plot_data_at_time_over_area(
#     time=x_0.date_time,
#     x_interval=x_interval,
#     y_interval=y_interval,
#     return_ax=True,
#     colorbar=False
# # )
# ax = planner_hycom.plot_reachability_snapshot(rel_time_in_seconds = 0, granularity_in_h = 5,
#                                    alpha_color = 1, time_to_reach=True,
#                                    return_ax = True, fig_size_inches=(12, 12),
#                                    input_ax = ax, plot_in_h = True, display_colorbar=True, mask_above_zero=True)
# ax = arena.plot_all_on_map(
#     problem= problem,
#     margin = 1
# )
# plt.show()
