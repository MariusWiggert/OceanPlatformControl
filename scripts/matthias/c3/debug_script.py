import os
import datetime
import logging
import random

import matplotlib.pyplot as plt
import numpy as np

import ocean_navigation_simulator
from ocean_navigation_simulator.problem_factories.SeaweedMissionGenerator import (
    SeaweedMissionGenerator,
)
from ocean_navigation_simulator.utils.misc import get_c3
from ocean_navigation_simulator.utils.misc import set_arena_loggers
from ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner import (
    HJBSeaweed2DPlanner,
)
from ocean_navigation_simulator.environment.SeaweedProblem import SeaweedProblem
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.problem_factories.Constructor import Constructor
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.utils import units

import os
import pickle
import logging
import datetime


import contextlib
import shutil
import time
import wandb


c3 = get_c3()

run = c3.OceanSimRun.get(
    "umax_0.3_30d_HC_copernicus_mission_nr_1298_SeaweedHJController_deg_HC_50_affine_bc_1_NoObserver"
)


"""Main function that takes in a spec and runs the simulator."""

# ensure we have all necessary data to run
this = run.get(
    "mission.missionConfig, mission.experiment.timeout_in_sec,"
    + "mission.experiment.arenaConfig, mission.experiment.objectiveConfig,"
    + "controllerSetting.ctrlConfig, observerSetting.observerConfig"
)
this

set_arena_loggers(logging.INFO)

# wandb.join()
# wandb.finish()


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


try:
    # ensure we have all necessary data to run
    this = this.get(
        "mission.missionConfig, mission.experiment.timeout_in_sec,"
        + "mission.experiment.arenaConfig, mission.experiment.objectiveConfig,"
        + "controllerSetting.ctrlConfig, observerSetting.observerConfig"
    )

    # # Step 0: check if mission is ready_to_run before running
    # if this.mission.get("status").status != 'ready_to_run':
    #     oceanSimResult = c3.OceanSimResult(**{
    #         'error_message': 'Mission status is not ready_to_run, run feasibility check first or manually set status.',
    #     })
    #     #####new_osr = c3.OceanSimRun(**{'id': this.id, 'status': 'run_failed', 'oceanSimResult': oceanSimResult})
    #     #new_osr.merge()
    #     return #new_osr

    # update the entry while it's running
    ##new_osr = c3.OceanSimRun(**{"id": this.id, "status": "running_sim"})
    # new_osr.merge()

    # set download directories (ignore set ones in arenaConfig)
    arenaConfig = this.mission.experiment.arenaConfig

    ### Prepare data for WandB
    ctrlConfig = this.controllerSetting.ctrlConfig
    # ctrlConfig["discount_factor_tau"]=500
    missionConfig = this.mission.missionConfig
    observerConfig = this.observerSetting.observerConfig

    # Prepare variables for run naming

    # Planning horizon in days
    T_in_days = ctrlConfig["T_goal_in_seconds"] / (24 * 3600)

    # Prepare string whether we use only HC or FC/HC
    if arenaConfig["ocean_dict"]["forecast"] is not None:
        data_sources = "FC_HC"
    else:
        data_sources = "HC"

    if (
        ctrlConfig["ctrl_name"]
        == "ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner.HJBSeaweed2DPlanner"
    ):
        ctrl_name = "HJ"
    else:
        ctrl_name = "undefined"

    umax = arenaConfig["platform_dict"]["u_max_in_mps"]
    deg_around_xt_xT_box = ctrlConfig["deg_around_xt_xT_box"]
    grid_res = ctrlConfig["grid_res"]

    # if not ctrlConfig.get("precomputation", False):
    # Log metrics in WandB
    os.environ["WANDB_API_KEY"] = "4c142c345dfc64f3c73aa1b2834989c7eb91efbe"

    # Randomly delay some runs in order stay within wandb request limits (200 per minute)
    # Generate a random delay - default 0
    delay = random.randint(0, ctrlConfig.get("wandb_delay", 0))
    # Pause the execution of code for the generated random delay value
    time.sleep(delay)

    # wandb.init(
    #     # Set the project where this run will be logged
    #     project="Long Horizon Seaweed Maximization",
    #     entity="matthiaskiller",
    #     id=this.id,
    #     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #     name=f"{ctrl_name}_{data_sources}_days_{T_in_days}_u_{umax}_deg_{deg_around_xt_xT_box}_res_{grid_res}_id_{this.id}",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "missionConfig": missionConfig,
    #         "arenaConfig": arenaConfig,
    #         "ctrlConfig": ctrlConfig,
    #         "observerConfig": observerConfig,
    #         "mission.id": this.mission.id,
    #         "experiment.id": this.mission.experiment.id,
    #         # "path_to_local_data": user_param["metrics_dir"],
    #     },
    #     settings=wandb.Settings(start_method="fork"),
    # )

    # create strings for all files and external directories where to save results
    # Set up file paths and download folders
    temp_folder = "temp/debug/"
    # create the folder if it doesn't exist yet
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
    traj_file_name = this.id + ".obj"
    extDir = "ocean_sim_run_results/" + this.mission.experiment.id + "/OceanSimRuns/" + this.id

    # Get and set the correct path to the nutrient and monthly average files (for c3) - !! CONFIG gets overwritten!!
    filepath = ocean_navigation_simulator.__file__
    module_path = os.path.dirname(filepath)
    nutrient_path = module_path + "/package_data/nutrients/"
    seaweed_maps_path = module_path + "/package_data/seaweed_growth_maps/"
    averages_path = module_path + "/package_data/monthly_averages/"

    ctrlConfig["seaweed_precomputation_folder"] = seaweed_maps_path

    arenaConfig["seaweed_dict"]["hindcast"]["source_settings"]["filepath"] = nutrient_path

    arenaConfig["timeout"] = this.mission.experiment.timeout_in_sec
    to_download_forecast_files = False

    # for hindcast
    arenaConfig["ocean_dict"]["hindcast"]["source_settings"][
        "folder"
    ] = "temp/debug/hindcast_files/"
    # For forecast
    if arenaConfig["ocean_dict"]["forecast"] is not None:
        arenaConfig["ocean_dict"]["forecast"]["source_settings"][
            "folder"
        ] = "temp/debug/forecast_files/"
        to_download_forecast_files = (
            arenaConfig["ocean_dict"]["forecast"]["source"] == "forecast_files"
            or arenaConfig["ocean_dict"]["forecast"]["source"] == "hindcast_as_forecast_files"
        )

    # For average
    if arenaConfig["ocean_dict"].get("average", None) is not None:
        arenaConfig["ocean_dict"]["average"]["source_settings"]["folder"] = averages_path

    # prepping the file download
    point_to_check = SpatioTemporalPoint.from_dict(this.mission.missionConfig["x_0"][0])
    t_interval = [
        point_to_check.date_time,
        point_to_check.date_time
        + datetime.timedelta(
            seconds=this.mission.experiment.timeout_in_sec
            + arenaConfig["casadi_cache_dict"]["time_around_x_t"]
            + 7200
        ),
    ]

    arena, controller = None, None

    if (
        arenaConfig["ocean_dict"]["forecast"] is not None
        and arenaConfig["ocean_dict"]["forecast"]["source"] == "hindcast_as_forecast_files"
    ):
        t_interval_adapted = [
            t_interval[0] - datetime.timedelta(days=2),
            t_interval[1]
            + datetime.timedelta(
                days=arenaConfig["ocean_dict"]["forecast"].get("forecast_length_in_days", 5)
            ),
        ]
    else:
        t_interval_adapted = t_interval

    # with (
    #     ArenaFactory.download_files(
    #         config=arenaConfig,
    #         type="hindcast",
    #         t_interval=t_interval,
    #         c3=c3,
    #         points=[point_to_check.to_spatial_point()],
    #         keep_newest_days=arenaConfig["ocean_dict"]["keep_newest_days"],
    #     ) as download_hindcast_files_to_local,
    #     ArenaFactory.download_files(
    #         config=arenaConfig,
    #         type="forecast",
    #         t_interval=t_interval_adapted,
    #         c3=c3,
    #         points=[point_to_check.to_spatial_point()],
    #         keep_newest_days=arenaConfig["ocean_dict"]["keep_newest_days"],
    #     )
    #     if to_download_forecast_files
    #     else dummy_context_mgr() as download_forecast_files_to_local,
    # ):
    # Step 0: Create Constructor object which contains arena, problem, controller and observer
    constructor = Constructor(
        arena_conf=arenaConfig,
        mission_conf=this.mission.missionConfig,
        objective_conf=this.mission.experiment.objectiveConfig,
        ctrl_conf=this.controllerSetting.ctrlConfig,
        observer_conf=this.observerSetting.observerConfig,
        c3=c3,
    )

    # Step 1.1 Retrieve problem
    problem = constructor.problem

    # Step 1.2: Retrieve arena
    arena = constructor.arena
    observation = arena.reset(platform_state=problem.start_state)
    problem_status = arena.problem_status(problem=problem)

    # Step 2: Retrieve Controller
    controller = constructor.controller

    # Step 3: Retrieve observer
    observer = constructor.observer
    observer.observe(observation)
    observation.forecast_data_source = observer

    # Step 4: Run Arena
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

    # new_osr.status = "finished_running"
    # new_osr.terminationReason = arena.problem_status_text(problem_status)
    # if arena.problem_status_text(problem_status) == "Success":
    #     #new_osr.T_arrival_time = (
    #     #    arena.state_trajectory[-1, 2] - arena.state_trajectory[0, 2]
    #     #) / 3600
    logged_error_message = None

    # if not ctrlConfig.get("precomputation", False):

    # Step 4: create the OceanSimResult object with the files and upload it
    if arena is not None:
        # Create a large dict with all trajectory data
        trajs_dict = {
            "sim_traj": arena.state_trajectory,
            "sim_ctrl": arena.action_trajectory,
        }
        if controller is not None:
            if len(controller.planned_trajs) > 0:
                trajs_dict["plans"] = controller.planned_trajs

        with open(temp_folder + traj_file_name, "wb") as traj_file:
            pickle.dump(trajs_dict, traj_file)

        # value_fct_file_name = this.id + "_value_fct.obj"
        # reach_times_file_name = this.id + "_reach_times.obj"

        # with open(temp_folder + value_fct_file_name, "wb") as value_fct_file:
        #     pickle.dump(np.array(constructor.controller.all_values), value_fct_file)

        # with open(temp_folder + reach_times_file_name, "wb") as reach_times_file:
        #     pickle.dump(
        #         np.array(constructor.controller.reach_times), reach_times_file
        #     )

        # Step 4.3 upload the traj_file_name together with the log_file_name to blob storage
        # upload log file
        # new_osr.oceanSimResult = c3.OceanSimResult()
        # if os.path.exists(temp_folder + log_file_name):
        #     c3.Client.uploadLocalClientFiles(
        #         temp_folder + log_file_name,
        #         extDir + "/logs",
        #         {"peekForMetadata": True},
        #     )
        #     log_file = c3.File(
        #         **{"url": extDir + "/logs/" + log_file_name}
        #     ).readMetadata()
        #     #new_osr.oceanSimResult.log_file = log_file
        #     os.remove(temp_folder + log_file_name)

        # if os.path.exists(temp_folder + traj_file_name):
        #     # upload traj file
        #     c3.Client.uploadLocalClientFiles(
        #         temp_folder + traj_file_name,
        #         extDir + "/trajs",
        #         {"peekForMetadata": True},
        #     )
        #     trajs_file = c3.File(
        #         **{"url": extDir + "/trajs/" + traj_file_name}
        #     ).readMetadata()
        #     #new_osr.oceanSimResult.trajectories = trajs_file
        #     os.remove(temp_folder + traj_file_name)

        # log shortest distance to target
        # if this.mission.experiment.objectiveConfig["type"] == "nav":
        #     new_osr.minDistanceToTarget = arena.final_distance_to_target(
        #         problem=problem
        #     )

    # wandb.summary["seaweed_mass_start"] = arena.state_trajectory[0, 4]
    # wandb.summary["seaweed_mass_end"] = arena.state_trajectory[-1, 4]
    # wandb.summary["seaweed_mass_average"] = np.mean(arena.state_trajectory[:, 4], axis=0)
    # wandb.summary["termination_reason"] = arena.problem_status_text(problem_status)

    # # Log state_trajectory as table/dataframe
    # wandb.log(
    #     {"data_to_plot": wandb.Table(dataframe=arena.get_plot_data_for_wandb())},
    #     commit=False,
    # )

    # log trajectory plot on seaweed as .jpg and .svg
    ax = arena.plot_all_on_map(problem=problem, background="seaweed", return_ax=True)
    fig = ax.get_figure()
    fig = plt.figure(fig)

    plt.savefig(temp_folder + "seaweed_trajectory_on_map.svg")
    plt.savefig(temp_folder + "seaweed_trajectory_on_map.jpg", dpi=80)

    # log trajectory plot on currents as .jpg and .svg
    ax = arena.plot_all_on_map(problem=problem, background="current", return_ax=True)
    fig = ax.get_figure()
    fig = plt.figure(fig)

    plt.savefig(temp_folder + "current_trajectory_on_map.svg")
    plt.savefig(temp_folder + "current_trajectory_on_map.jpg", dpi=80)

    # image_seaweed = wandb.Image(
    #     temp_folder + "seaweed_trajectory_on_map.jpg",
    #     caption="Seaweed trajectory on map",
    # )

    # image_currents = wandb.Image(
    #     temp_folder + "current_trajectory_on_map.jpg",
    #     caption="Current trajectory on map",
    # )

    # wandb.log({"Seaweed trajectory on map": image_seaweed}, commit=False)
    # wandb.log({"Current trajectory on map": image_currents}, commit=True)

    # # log trajectory animations
    # arena.animate_trajectory(
    #     margin=0.25,
    #     problem=problem,
    #     temporal_resolution=14400,
    #     background="current",
    #     output=f"{temp_folder}platform_trajectories_currents.gif",
    #     fps=6,
    # )
    # arena.animate_trajectory(
    #     margin=0.25,
    #     problem=problem,
    #     temporal_resolution=14400,
    #     background="seaweed",
    #     output=f"{temp_folder}platform_trajectories_seaweed.gif",
    #     fps=6,
    # )

    # ### Save files to c3 and add link to WandB
    # traj_seaweed_plot_file_name = "seaweed_trajectory_on_map.svg"
    # traj_currents_plot_file_name = "current_trajectory_on_map.svg"
    # traj_seaweed_animation_file_name = "platform_trajectories_seaweed.gif"
    # traj_currents_animation_file_name = "platform_trajectories_currents.gif"

    # Save files on c3/ Azure blob storage
    # c3.Client.uploadLocalClientFiles(
    #     temp_folder + value_fct_file_name,
    #     extDir + "/logs",
    #     {"peekForMetadata": True},
    # )
    # value_fct_file = c3.File(
    #     **{"url": extDir + "/logs/" + value_fct_file_name}
    # ).readMetadata()
    # #new_osr.oceanSimResult.value_fct_file = value_fct_file
    # os.remove(temp_folder + value_fct_file_name)

    # c3.Client.uploadLocalClientFiles(
    #     temp_folder + reach_times_file_name,
    #     extDir + "/logs",
    #     {"peekForMetadata": True},
    # )
    # reach_times_file = c3.File(
    #     **{"url": extDir + "/logs/" + reach_times_file_name}
    # ).readMetadata()
    # #new_osr.oceanSimResult.reach_times_file = reach_times_file
    # os.remove(temp_folder + reach_times_file_name)

    # c3.Client.uploadLocalClientFiles(
    #     temp_folder + traj_seaweed_plot_file_name,
    #     extDir + "/logs",
    #     {"peekForMetadata": True},
    # )
    # traj_seaweed_plot_file = c3.File(
    #     **{"url": extDir + "/logs/" + traj_seaweed_plot_file_name}
    # ).readMetadata()
    # #new_osr.oceanSimResult.traj_seaweed_plot_file = traj_seaweed_plot_file
    # os.remove(temp_folder + traj_seaweed_plot_file_name)

    # c3.Client.uploadLocalClientFiles(
    #     temp_folder + traj_currents_plot_file_name,
    #     extDir + "/logs",
    #     {"peekForMetadata": True},
    # )
    # traj_currents_plot_file = c3.File(
    #     **{"url": extDir + "/logs/" + traj_currents_plot_file_name}
    # ).readMetadata()
    # #new_osr.oceanSimResult.traj_currents_plot_file = traj_currents_plot_file
    # os.remove(temp_folder + traj_currents_plot_file_name)

    # c3.Client.uploadLocalClientFiles(
    #     temp_folder + traj_currents_animation_file_name,
    #     extDir + "/logs",
    #     {"peekForMetadata": True},
    # )
    # traj_currents_animation_file = c3.File(
    #     **{"url": extDir + "/logs/" + traj_currents_animation_file_name}
    # ).readMetadata()
    # #new_osr.oceanSimResult.traj_currents_animation_file = (
    #     traj_currents_animation_file
    # )
    # os.remove(temp_folder + traj_currents_animation_file_name)

    # c3.Client.uploadLocalClientFiles(
    #     temp_folder + traj_seaweed_animation_file_name,
    #     extDir + "/logs",
    #     {"peekForMetadata": True},
    # )
    # traj_seaweed_animation_file = c3.File(
    #     **{"url": extDir + "/logs/" + traj_seaweed_animation_file_name}
    # ).readMetadata()
    # #new_osr.oceanSimResult.traj_seaweed_animation_file = traj_seaweed_animation_file
    # os.remove(temp_folder + traj_seaweed_animation_file_name)

    # Log file urls (links to azure) to wandb summary
    # wandb.summary["value_fct_array"] = value_fct_file.generatePresignedUrl(
    #     "GET", "600d"
    # )

    # wandb.summary["reach_times_array"] = reach_times_file.generatePresignedUrl(
    #     "GET", "600d"
    # )

    # wandb.summary[
    #     "traj_seaweed_plot_svg"
    # ] = traj_seaweed_plot_file.generatePresignedUrl("GET", "600d")
    # wandb.summary[
    #     "traj_current_plot_svg"
    # ] = traj_currents_plot_file.generatePresignedUrl("GET", "600d")
    # wandb.summary[
    #     "traj_currents_animation"
    # ] = traj_currents_animation_file.generatePresignedUrl("GET", "600d")
    # wandb.summary[
    #     "traj_seaweed_animation"
    # ] = traj_seaweed_animation_file.generatePresignedUrl("GET", "600d")

    # if os.path.exists(temp_folder + log_file_name):
    #     wandb.summary["log_file"] = log_file.generatePresignedUrl("GET", "600d")

# wandb.finish()


except BaseException as e:
    # if we crash we should upload that to the table for debugging
    print("Error Message: ", e)
    # new_osr.terminationReason = "run_failed"
    # new_osr.status = "run_failed"
    logged_error_message = e
    # if wandb.run is not None:
    #     wandb.summary["caught_error"] = True
    #     # wandb.summary["caught_error_message"] = e
    #     wandb.finish()

    raise e

# if new_osr.oceanSimResult is None:
#     new_osr.oceanSimResult = c3.OceanSimResult(
#         **{"error_message": logged_error_message}
#     )
# else:
# new_osr.oceanSimResult.error_message = logged_error_message

# new_osr.merge()
# try:
#     shutil.rmtree(temp_folder)
# except BaseException as e:
#     print(e)

# return #new_osr.oceanSimResult
