# This script shows how to set-up and run the simplex noise ocean field.
# Note the simplex parameters are tuned for Region 1
import datetime

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

# Define the OceanCurrent Source with a base HC and a Noise to add on top
arena_config = {
    "casadi_cache_dict": {"deg_around_x_t": 0.5, "time_around_x_t": 36000.0},
    "platform_dict": {
        "battery_cap_in_wh": 400.0,
        "u_max_in_mps": 0.1,
        "motor_efficiency": 1.0,
        "solar_panel_size": 0.5,
        "solar_efficiency": 0.2,
        "drag_factor": 675.0,
        "dt_in_s": 1800.0,
    },
    "use_geographic_coordinate_system": True,
    "spatial_boundary": None,
    "timeout": 432000,
    "ocean_dict": {
        "region": "Region 1",
        "hindcast": {
            "field": "OceanCurrents",
            "source": "generative_noise",
            "base_source": {
                "source": "hindcast_files",
                "source_settings": {
                    "local": True,
                    "folder": "data/generative_noise_tutorial/",
                    "source": "copernicus",
                    "type": "hindcast",
                },
            },
            "noise_source": {
                "source": "simplex_noise",
                "source_settings": {
                    "seed": 151,  # This is the seed for the simplex noise process
                    "params_path": "ocean_navigation_simulator/generative_error_model/models/tuned_2d_forecast_variogram_area1_[5.0, 1.0]_False_True.npy",
                    "scale_noise": 1,
                },
            },
        },
        "forecast": None,
    },
    "solar_dict": {"hindcast": None, "forecast": None},
    "seaweed_dict": {"hindcast": None, "forecast": None},
}

t_interval = [datetime.datetime(2022, 10, 1, 13, 30, 0), datetime.datetime(2022, 10, 1, 20, 30, 0)]

# download files if not there yet
ArenaFactory.download_required_files(
    archive_source="copernicus",
    archive_type="forecast",  # should be hindcast once that works on C3
    region="Region 1",
    download_folder="data/generative_noise_tutorial/",
    t_interval=t_interval,
)
# create the arena
arena = ArenaFactory.create(scenario_config=arena_config)

#%% define intervals for plotting
x_interval = [-140, -135]
y_interval = [20, 25]
t_plot = datetime.datetime(2022, 10, 4, 12, 00, 00, tzinfo=datetime.timezone.utc)

#%% plot default
arena.ocean_field.hindcast_data_source.plot_noise_at_time_over_area(
    time=t_plot, x_interval=x_interval, y_interval=y_interval
)
#%% loop over possible seed_integers and plot all
for i in range(151, 152):
    print(i)
    arena.ocean_field.hindcast_data_source.set_noise_seed(seed_integer=i)
    arena.ocean_field.hindcast_data_source.plot_noise_at_time_over_area(
        time=t_plot, x_interval=y_interval, y_interval=x_interval
    )
