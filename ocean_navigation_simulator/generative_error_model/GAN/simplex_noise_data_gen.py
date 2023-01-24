from ocean_navigation_simulator.generative_error_model.models.OceanCurrentNoiseField import OceanCurrentNoiseField
from ocean_navigation_simulator.generative_error_model.GAN.data_preprocessing import area_handler

import numpy as np
import datetime
import os


# TSN parameters
params_path = "../../../data/drifter_data/variogram_params/tuned_2d_forecast_variogram_area1_edited2.npy"
# print(np.load(params_path, allow_pickle=True))

noise_field = OceanCurrentNoiseField.load_config_from_file(params_path)
rng = np.random.default_rng(123)
noise_field.reset(rng)

# define the spatial and temporal range
lon_range, lat_range = area_handler("area1")
t_range = [datetime.datetime(2022, 5, 2, 12, 30, 0),
           datetime.datetime(2022, 5, 11, 12, 30, 0)]

# output
datasets = {"train_val": 139, "test": 16}
root = "../../../data/drifter_data/synthetic_data"
for dataset, num_files in datasets.items():
    dir = os.path.join(root, dataset)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in range(num_files):

        # get the noise
        noise = noise_field.get_noise_from_ranges(lon_range, lat_range, t_range)

        # # save noise as nc file
        # file_name = f"synth_data_{lon_range}_{lat_range}_{t_range}.nc"
        # noise.to_netcdf(os.path.join(dir, file_name))

        # save as npy file
        file_name = f"synth_data_{lon_range}_{lat_range}_[{t_range[0]}_{t_range[1]}]"
        data_u, data_v = noise["water_u"].values, noise["water_v"].values
        data_u, data_v = data_u[:, np.newaxis, :, :], data_v[:, np.newaxis, :, :]
        data = np.concatenate([data_u, data_v], axis=1)
        np.save(os.path.join(dir, file_name), data)

        # advance time window by 9 days
        t_range = [time + datetime.timedelta(days=9) for time in t_range]