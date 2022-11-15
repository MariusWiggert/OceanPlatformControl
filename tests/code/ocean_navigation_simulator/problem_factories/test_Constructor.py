import os

import datetime
import time

import yaml

from ocean_navigation_simulator.problem_factories.Constructor import Constructor

import pytest


@pytest.mark.training
def test_Constructor():
    """Test of E2E Test of the Constructor class. It runs a evaluation script and ensures no errors occur"""

    print("This test might take several minutes")

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

    # observer_config = {
    #     "observer": None
    # }


    x_0 = {
        "lon": -82.5,
        "lat": 23.7,
        "date_time": "2021-11-24 23:12:00.004573 +0000",
    }

    x_T = {"lon": -80.3, "lat": 24.6}

    with open(f"config/arena/gulf_of_mexico_HYCOM_hindcast_local.yaml") as f:
        arena_config = yaml.load(f, Loader=yaml.FullLoader)

    mission_config = {
        "x_0": [x_0],
        "x_T": x_T,
        "target_radius": 0.1,
        "timeout": datetime.timedelta(days=5),
        "seed": 12344093,
    }

    ctrl_config = {
        "ctrl_name": "ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner",
        "replan_on_new_fmrc": True,
        "replan_every_X_seconds": False,
        "direction": "backward",
        "n_time_vector": 200,
        # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
        "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
        "accuracy": "high",
        "artificial_dissipation_scheme": "local_local",
        "T_goal_in_seconds": 3600 * 24 * 5,
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


    # Step 0: Create Constructor object which contains arena, problem, controller and observer
    constructor = Constructor(
        arena_conf=arena_config,
        mission_conf=mission_config,
        objective="nav",
        ctrl_conf=ctrl_config,
        observer_conf=observer_config,
    )


    # Step 1.1 Retrieve problem
    problem = constructor.problem

    # Step 1.2: Retrieve arena
    arena = constructor.arena
    observation = arena.reset(platform_state=problem.start_state)
    problem_status = arena.problem_status(problem=problem)


    # Step 2: Retrieve controller
    controller = constructor.controller
    action = controller.get_action(observation=observation)

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

    # Generate trajectory animation
    arena.animate_trajectory(problem=problem, temporal_resolution=7200,output="trajectory.mp4")


    # Test if animation is saved
    assert os.path.isfile('generated_media/trajectory.mp4') is True, "Trajectory animation was not saved properly."

    # Remove animation
    os.remove('generated_media/trajectory.mp4')