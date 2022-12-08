from ocean_navigation_simulator.generative_error_model.GAN.data_preprocessing import save_sparse_as_npy, area_handler
from ocean_navigation_simulator.generative_error_model.utils import get_datetime_from_file_name, datetime2str

import os
import numpy as np
import glob
import pandas as pd
import xarray as xr
import datetime
import itertools


class ConvertToError:
    """GAN predicts in current space, therefore the FC has to be subtracted from the predictions. Additionally,
    to make the data easier to handle, it is converted to a NetCDF file."""

    def __init__(self, predictions_dir: str, ground_truth_dir: str):
        self.samples_dir = predictions_dir
        self.ground_truth_dir = ground_truth_dir

        # data saved from test function of GAN
        fc_files = get_file_path_list(predictions_dir, "input")
        pred_files = get_file_path_list(predictions_dir, "output")

        self.fc_data = [np.load(file_path, mmap_mode="r+", allow_pickle=True) for file_path in fc_files]
        self.predictions = [np.load(file_path, mmap_mode="r+", allow_pickle=True) for file_path in pred_files]

    def get_files_start_end_date_time(self):
        """Uses ground_truth files to determine time range of GAN samples."""

        gt_files = sorted(os.listdir(self.ground_truth_dir))
        start_datetime = get_datetime_from_file_name(gt_files[0])
        end_datetime = get_datetime_from_file_name(gt_files[-1]) + datetime.timedelta(days=8)
        return start_datetime, end_datetime

    def get_max_volume_as_nc(self):
        """Takes predictions and inputs from GAN test function, calculates the error (pred - fc) and converts to
        an NetCDF file.

        The GAN only uses the first 8 days of data of each file. In time these samples though still overlap.
        Therefore to produce a NetCDF file which has a time axis, one can only use the first days prediction
        for every file and the entire prediction of the last file."""

        # get number of files
        num_files = len(os.listdir(self.ground_truth_dir))
        final_time_steps = (num_files-1)*24 + 8*24

        final_data = np.zeros((final_time_steps, 2, 256, 256))
        print(final_data.shape)
        for i in range(len(self.fc_data)-1):
            final_data[i*24:(i+1)*24, :, :, :] = self.predictions[i][:24] - self.fc_data[i][:24]
        final_data[len(self.fc_data):len(self.fc_data) + 8*24, :, :, :] = self.predictions[-1] - self.fc_data[-1]

        # get axes for nc file
        start_datetime, end_datetime = self.get_files_start_end_date_time()
        start_end_timedelta = end_datetime - start_datetime
        hour_range = start_end_timedelta.days*24 + start_end_timedelta.seconds//3600
        t_axis, lat_axis, lon_axis = self.get_axes(start_datetime, hour_range, "area1")
        return self._create_xarray(final_data, lon_axis, lat_axis, t_axis)

    def get_individual_as_nc(self, output_dir: str, type: str="error", duplicate: bool = False):
        """Take a single prediction/forecast file and converts it to an nc file."""

        gt_files = sorted(os.listdir(self.ground_truth_dir))
        for i in range(len(self.fc_data)):
            # get axes of data
            start_datetime = get_datetime_from_file_name(gt_files[i])
            hour_range = 8 * 24
            t_axis, lat_axis, lon_axis = self.get_axes(start_datetime, hour_range, "area1")

            # check if file already exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_name = f"GAN_pred_lon_{lon_axis[0]},{lon_axis[-1]}_lat_{lat_axis[0]},{lat_axis[-1]}_" \
                        f"time_{datetime2str(t_axis[0])},{datetime2str(t_axis[-1])}"
            file_path = f"{output_dir}/{file_name}.nc"
            if os.path.exists(file_path):
                print(f"File: {file_name} already exists!")
                if duplicate:
                    index = 1
                    while os.path.exists(file_path):
                        file_path = ".".join(file_path.split(".")[:-1]) + str(index) + ".nc"
                        index += 1
                else:
                    continue

            if type == "error":
                data = self.predictions[i] - self.fc_data[i]
            elif type == "current":
                data = self.predictions[i]
            elif type == "forecast":
                data = self.fc_data[i]
            nc = self._create_xarray(data, lon_axis, lat_axis, t_axis)
            nc.to_netcdf(file_path)

    def generate_data_for_evaluation(self):
        """Formats data for both RMSE and Vector Correlation comparison, as well as for Variogram
        computation."""

        pass

    @staticmethod
    def get_axes(start_datetime: datetime.datetime, hour_range: int, area: str):
        """Returns axes needed for xarray object."""

        t_axis = [start_datetime + datetime.timedelta(hours=num_hours) for num_hours in range(hour_range)]
        lon_range, lat_range = area_handler(area)
        lon_axis = np.linspace(lon_range[0], lon_range[1], 256, endpoint=True)
        lat_axis = np.linspace(lat_range[0], lat_range[1], 256, endpoint=True)
        return t_axis, lat_axis, lon_axis

    @staticmethod
    def _create_xarray(
        data: np.ndarray, lon_axis: np.ndarray, lat_axis: np.ndarray, t_axis: list) -> xr.Dataset:

        ds = xr.Dataset(
            data_vars=dict(
                water_u=(["time", "lat", "lon"], data[:, 0, :, :]),
                water_v=(["time", "lat", "lon"], data[:, 1, :, :]),
            ),
            coords=dict(
                time=t_axis,
                lat=lat_axis,
                lon=lon_axis
            ),
            attrs=dict(description="An ocean current error sample over time and space."),
        )
        return ds


def get_sparse_from_nc(nc_file: str, num_samples: int = 30000):
    """Takes an nc file and produces sparse samples. Samples are saved such that
    they can be used directly with variogram code."""

    data = xr.load_dataset(nc_file)
    lon_len = len(data["lon"].values)
    lat_len = len(data["lat"].values)
    time_len = len(data["time"].values)
    total_len = lon_len * lat_len * time_len

    idx = np.random.choice(list(range(total_len)), size=num_samples)

    lon = data["lon"].values.reshape(-1)
    lat = data["lat"].values.reshape(-1)
    time = data["time"].values.reshape(-1)

    axes = np.array(list(itertools.product(time, lat, lon)))[idx]
    time = axes[:, 0]
    lat = axes[:, 1]
    lon = axes[:, 2]
    u_error = data["water_u"].values.reshape(-1)[idx]
    v_error = data["water_v"].values.reshape(-1)[idx]

    data = pd.DataFrame({"lon": lon,
                         "lat": lat,
                         "time": time,
                         "u_error": u_error,
                         "v_error": v_error})
    return data


def get_file_path_list(dir: str, part_string: str):
    return sorted(glob.glob(f"{dir}/{part_string}*.npy"))


def main():
    predictions_dir = "data/drifter_data/GAN_samples/2022-11-28_20:38:37"
    ground_truth_dir = "data/drifter_data/buoy_preprocessed_test/area1"
    eval = ConvertToError(predictions_dir, ground_truth_dir)
    eval.get_individual_as_nc("data/drifter_data/GAN_nc")


if __name__ == "__main__":
    main()
