import datetime

from ocean_navigation_simulator.reinforcement_learning.missions.MissionGenerator import (
    MissionGenerator,
)
from ocean_navigation_simulator.utils import units

problem_factory = MissionGenerator(
    config={
        "scenario_file": "config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
        ##### Target Sampling #####
        # HYCOM HC: lon [-98.0,-76.4000244140625], lat[18.1200008392334,31.92000007629394]
        # Copernicus FC: lon: [-98.0, -76.416664], lat: [18.083334, 30.0]
        # Combined: [-98.0, -76.416664], [18.1200008392334, 30.0]
        "x_range": [units.Distance(deg=-95.9), units.Distance(deg=-78.52)],
        "y_range": [units.Distance(deg=20.22), units.Distance(deg=27.9)],
        "t_range": [
            # Copernicus FC: 2022-04 until today, HYCOM Hindcast: 2021-09 until today
            datetime.datetime(year=2022, month=4, day=8, tzinfo=datetime.timezone.utc),
            datetime.datetime(year=2022, month=10, day=9, tzinfo=datetime.timezone.utc),
        ],
        "problem_timeout": datetime.timedelta(hours=150),
        "target_distance_from_land": 0.5,
        "problem_target_radius": 0.1,
        ##### HJ Planner #####
        "hj_specific_settings": {
            # 'grid_res' has to be smaller than target_radius to prevent hj_solver errors
            "grid_res": 0.04,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
            "direction": "multi-time-reach-back",
            "n_time_vector": 199,  # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
            "accuracy": "high",
            "artificial_dissipation_scheme": "local_local",
            "run_without_x_T": True,
            "progress_bar": False,
        },
        "hj_planner_box": 2.0,
        ##### Start Sampling #####
        "feasible_missions_per_target": 8,
        "random_missions_per_target": 8,
        "min_distance_from_hj_frame": 0.5,
        "min_distance_from_land": 0.5,
        "feasible_mission_time": [datetime.timedelta(hours=20), datetime.timedelta(hours=120)],
        "random_min_distance_from_target": 0.5,
        ##### Actions #####
        "plot_batch": True,
        "animate_batch": False,
        "cache_forecast": True,
        "cache_hindcast": True,
    },
    verbose=5,
)
problems = problem_factory.generate_batch()
