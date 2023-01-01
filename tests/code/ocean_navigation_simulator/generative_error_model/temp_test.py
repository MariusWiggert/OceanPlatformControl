from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import (
    GroundTruthFromNoise,
    HindcastFileSource,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

import datetime

# define intervals
lon_interval = [-140 + 360, -120 + 360]
lat_interval = [20, 30]
t_interval = [datetime.datetime(2022, 10, 1, 12, 30, 0), datetime.datetime(2022, 10, 8, 12, 30, 0)]
target_folder = "data/hycom_hindcast_gen_noise_test/"

# TODO: Maybe implement a couple of new plotting functions to plot/animate purely the noise for a specific area?
# TODO: Go over this example, with noise it looks quite a lot different, maybe too much? Not sure...
# TODO: ultimately we want an ocean field to be as easy to instantiate as the others, directly from one dict, build the constructors to do that.
#%% download files if not there yet
ArenaFactory.download_required_files(
    archive_source="hycom",
    archive_type="hindcast",
    region="Region 1",
    download_folder=target_folder,
    t_interval=t_interval,
)
#%% Getting the GT data Source by adding Generative Noise
source_dict = {
    "field": "OceanCurrents",
    "source": "hindcast_files",
    "use_geographic_coordinate_system": True,
    "source_settings": {"folder": target_folder},
}
hindcast_data = HindcastFileSource(source_dict)

gt = GroundTruthFromNoise(
    seed=123,
    params_path="ocean_navigation_simulator/generative_error_model/models/"
    + "tuned_2d_forecast_variogram_area1_[5.0, 1.0]_False_True.npy",
    # + "tuned_2d_forecast_variogram_area1_edited.npy",
    hindcast_data_source=hindcast_data,
)

gt.plot_noise_at_time_over_area(
    time=t_interval[0], x_interval=lon_interval, y_interval=lat_interval
)
#%% plot comparison
# without noise
hindcast_data.plot_data_at_time_over_area(
    time=t_interval[0], x_interval=lon_interval, y_interval=lat_interval
)
#%% # with Noise
gt.plot_data_at_time_over_area(time=t_interval[0], x_interval=lon_interval, y_interval=lat_interval)
