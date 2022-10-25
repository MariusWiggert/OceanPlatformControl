from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import (
    GroundTruthFromNoise,
    ForecastFromHindcastSource,
    HindcastFileSource
)
import datetime

source_dict = {
    "field": "OceanCurrents",
    "source": "hindcast_files",
    "source_settings": {
        "folder": "data/drifter_data/hindcasts/area1/",
        "currents": "total"
    }
}

# define intervals
lon_interval = [-140, -120]
lat_interval = [20, 30]
t_interval = [datetime.datetime(2022, 10, 1, 12, 30, 0),
              datetime.datetime(2022, 10, 8, 12, 30, 0)]

hindcast_data = HindcastFileSource(source_dict)

gt = GroundTruthFromNoise(123,
                          "ocean_navigation_simulator/generative_error_model/models/" +
                          "tuned_2d_forecast_variogram_area1_[5.0, 1.0]_False_True.npy",
                          hindcast_data)

print(gt.get_data_over_area(lon_interval, lat_interval, t_interval))
