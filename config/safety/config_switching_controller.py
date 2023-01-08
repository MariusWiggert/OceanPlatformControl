specific_settings_navigation = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 1,
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
    # TODO: This is from the config. So we can perhaps pass the whole config and then pick what we need
    "platform_dict": None,
}

specific_settings_safety = {
    "filepath_distance_map": {
        "bathymetry": "data/bathymetry/bathymetry_distance_res_0.083_0.083_max.nc",
        "garbage": "data/garbage_patch/garbage_patch_distance_res_0.083_0.083_max.nc",
    }
}
specific_settings_switching = dict(
    safety_condition={"base_setting": "on", "area_type": ["bathymetry", "garbage"]},
    safe_distance={"bathymetry": 10, "garbage": 10},  # TODO: change to units,
    safety_controller="ocean_navigation_simulator.controllers.NaiveSafetyController.NaiveSafetyController(problem, self.specific_settings_safety)",
    navigation_controller="ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner(problem, self.specific_settings_navigation)",
)
