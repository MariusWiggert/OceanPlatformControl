import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from ocean_navigation_simulator.problem_factories.Constructor import (
    Constructor,
)
import pandas as pd
from ocean_navigation_simulator.problem_factories.Constructor import (
    Constructor,
)
from ocean_navigation_simulator.reinforcement_learning.missions.MissionGenerator import (
    MissionGenerator,
)
from ocean_navigation_simulator.reinforcement_learning.runners.GenerationRunner import (
    GenerationRunner,
)
from ocean_navigation_simulator.reinforcement_learning.missions.CachedNavigationProblem import (
    CachedNavigationProblem,
)
from ocean_navigation_simulator.utils.misc import set_arena_loggers
import yaml
import numpy as np
import argparse
import wandb
from pathlib import Path


def get_project_root() -> Path:
    # gets to OceanPlatformControl directory
    # to change if this file is another directory !!
    return Path(__file__).parent.parent.parent.parent


def get_config_dict(controller_name, user_param: dict):
    arena_config = {
        "casadi_cache_dict": {"deg_around_x_t": 2.0, "time_around_x_t": 432000},
        "timeout": user_param_dict["timeout_h"] * 3600,
        "platform_dict": {
            "battery_cap_in_wh": 400.0,
            "u_max_in_mps": 0.1,
            "motor_efficiency": 1.0,
            "solar_panel_size": 0.5,
            "solar_efficiency": 0.2,
            "drag_factor": 675.0,
            "dt_in_s": 600.0,
        },
        "use_geographic_coordinate_system": True,
        "spatial_boundary": None,
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
                    #  "region": "Region 1"
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
            #         # "region": "Region 1",
            #     },
            # },
        },
        "multi_agent_constraints": {
            "unit": "km",
            "communication_thrsld": 9,
            "epsilon_margin": 1,  # when add edges based on hysteresis
            "collision_thrsld": 0.2,
        },
    }
    NoObserver = {"observer": None}

    # Controller Configs
    HJMultiTimeConfig = {
        "replan_every_X_seconds": None,
        "replan_on_new_fmrc": True,
        "T_goal_in_seconds": 3600 * 24 * 3,  # 3d, 43200,     # 12h
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

    MultiAgentCtrlConfig = {
        "ctrl_name": "ocean_navigation_simulator.controllers.MultiAgentPlanner.MultiAgentPlanner",
        "high_level_ctrl": user_param[
            "controller"
        ],  # choose from hj_naive, flocking, reactive_control
        "unit": "km",
        "communication_thrsld": 9,
        "hj_specific_settings": HJMultiTimeConfig,
        "flocking_config": flockingConfig,
        "reactive_control_config": reactiveConfig,
    }
    objective_conf = {"type": "nav"}
    return arena_config, MultiAgentCtrlConfig, objective_conf, NoObserver


def run_mission(problem: CachedNavigationProblem, args, user_param: dict):
    arena_config, MultiAgentCtrlConfig, objective_conf, NoObserver = get_config_dict(
        controller_name=user_param["controller"],
        user_param=user_param,
    )
    mission_config = problem.to_c3_mission_config()
    # Step 0: Create Constructor object which contains arena, problem, controller and observer
    constructor = Constructor(
        arena_conf=arena_config,
        mission_conf=mission_config,
        objective_conf=objective_conf,
        ctrl_conf=MultiAgentCtrlConfig,  # here different controller configs can be put in
        observer_conf=NoObserver,  # here the other observers can also be put int
        download_files=True,
    )
    # Step 1.1 Retrieve problem
    problem = constructor.problem

    # Step 1.2: Retrieve arena
    arena = constructor.arena
    observation = arena.reset(platform_set=problem.start_state)
    problem_status = arena.problem_status(problem=problem)

    # Step 2: Retrieve Controller
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
    plt.savefig(f"{user_param['reachability_dir']}/_{idx_mission}.svg")
    # Step 3: Retrieve observer
    observer = constructor.observer

    # Step 4: Run closed-loop simulation
    ctrl_deviation_from_opt = []
    all_pltf_status = [0] * len(mission_config["x_0"])
    min_distances_to_target_over_mission = [np.inf] * len(mission_config["x_0"])
    pb_timeout_flag = -1
    while not any(status == pb_timeout_flag for status in problem_status):
        # Get action
        action, ctrl_correction = controller.get_action(observation=observation)
        ctrl_deviation_from_opt.append(ctrl_correction)
        # execute action
        observation = arena.step(action)
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

    # Plot useful metrics for multi-agent performance evaluation
    plt.clf()
    fig = arena.plot_all_network_analysis(
        xticks_temporal_res=12 * 3600
    )  # 8 hours interval for xticks
    plt.savefig(f"{user_param['graph_dir']}/network_prop_{idx_mission}.svg")
    plt.clf()
    arena.plot_distance_evolution_between_platforms()
    plt.savefig(f"{user_param['graph_dir']}/distance_evolution_{idx_mission}.svg")

    # Save animations for latter inspection
    arena.animate_trajectory(
        margin=0.25,
        problem=problem,
        temporal_resolution=7200,
        output=f"{user_param['animation_dir']}/trajectory_anim.gif",
        fps=10,
    )
    arena.animate_graph_net_trajectory(
        temporal_resolution=7200,
        # collision_communication_thrslds=(10, 50), (not specified take defaut one)
        plot_ax_ticks=True,
        output=f"{user_param['animation_dir']}/network_graph_trajectory_{idx_mission}.gif",
        fps=5,
    )

    # Log metrics in WandB
    wandb.init(
        # Set the project where this run will be logged
        project="Master Thesis",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{idx_mission}_{user_param['controller']}",
        # Track hyperparameters and run metadata
        config={
            "mission_config": mission_config,
            "arena_config": arena_config,
            "ctrl_config": MultiAgentCtrlConfig,
            "path_to_local_data": user_param["metrics_dir"],
        },
        entity="nhoischen",
    )
    metrics_dict = arena.save_metrics_to_log(
        all_pltf_status=all_pltf_status,
        min_distances_to_target=min_distances_to_target_over_mission,
        max_correction_from_opt_ctrl=ctrl_deviation_from_opt,
        filename=f"{user_param['metrics_dir']}/_{idx_mission}.log",
    )
    metrics_df = pd.DataFrame(data=metrics_dict, index=[0])
    wandb.log({"metrics": metrics_df})
    wandb.log({"data_to_plot": wandb.Table(dataframe=arena.get_plot_data_for_wandb())})
    wandb.finish()


if __name__ == "__main__":
    project_root_path = get_project_root()
    os.chdir(project_root_path)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ctrl",
        "--controller",
        help="controller name to run simulation",
        type=str,
        default="hj_naive",
    )
    parser.add_argument(
        "-filename_problems",
        "--filename_problems",
        help="path from ocean platform directory to where the problem is stored (provide as .csv file)",
        default="problemsGOM.csv",
    )
    parser.add_argument(
        "-timeout_h_buffer",
        "--timeout_h_buffer",
        help="timeout buffer in [h]",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-results_filename",
        "--results_filename",
        help="path from ocean platform directory to where the results will be saved",
        default="results",
    )
    args = parser.parse_args()
    # print(args.path_to_problems)
    # print(args.timeout_h)
    path_to_problems = os.path.join(project_root_path, "tmp/missions/" + args.filename_problems)
    path_to_results = os.path.join(project_root_path, "tmp/missions/" + args.results_filename)
    problems_df = pd.read_csv(path_to_problems)
    os.makedirs(f"{path_to_results}/{args.controller}/reachability_snapshots", exist_ok=True)
    os.makedirs(f"{path_to_results}/{args.controller}/metric_logs", exist_ok=True)
    os.makedirs(f"{path_to_results}/{args.controller}/animations", exist_ok=True)
    os.makedirs(f"{path_to_results}/{args.controller}/graph_analysis", exist_ok=True)
    for idx_mission in range(len(problems_df)):
        try:
            user_param_dict = {
                "reachability_dir": f"{path_to_results}/{args.controller}/reachability_snapshots",
                "metrics_dir": f"{path_to_results}/{args.controller}/metric_logs",
                "animation_dir": f"{path_to_results}/{args.controller}/animations",
                "graph_dir": f"{path_to_results}/{args.controller}/graph_analysis",
                "controller": args.controller,
                "timeout_h": problems_df.iloc[idx_mission]["ttr_in_h"] + args.timeout_h_buffer,
            }
            problem = CachedNavigationProblem.from_pandas_row(problems_df.iloc[idx_mission])
            print(f"---- RUN MISSION {idx_mission} ----")
            run_mission(problem=problem, args=args, user_param=user_param_dict)

        except Exception as e:
            print("an exception has occured for mission: ", idx_mission)
            log_directory = f"{path_to_results}/{args.controller}"
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)
            log_file = os.path.join(log_directory, "logfile.log")
            logging.basicConfig(
                format="%(asctime)s %(levelname)s %(message)s",
                filename=log_file,
                level=logging.ERROR,
                force=True,
            )
            logging.error(
                f"Error for mission index {idx_mission}: \n \
                            {e}"
            )
            continue
