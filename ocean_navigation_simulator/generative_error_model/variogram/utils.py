from ocean_navigation_simulator.generative_error_model.variogram.VisualizeVariogram import VisualizeVariogram
from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram
from typing import List
import pandas as pd
import numpy as np


def save_tuned_empirial_variogram_3d(vvis: VisualizeVariogram, view_range: List[int], file_path: str):
    """Convert hand-tuned variogram to dataframe and save."""

    lon_lag, lat_lag, time_lag = [], [], []
    u_semivariance, v_semivariance = [], []
    for lon in range(int(view_range[0]/vvis.variogram.res_tuple[0])):
        for lat in range(int(view_range[1]/vvis.variogram.res_tuple[1])):
            for time in range(int(view_range[2]/vvis.variogram.res_tuple[2])):
                u_semivariance.append(vvis.variogram.bins[lon, lat, time, 0])
                v_semivariance.append(vvis.variogram.bins[lon, lat, time, 1])
                lon_lag.append((lon+1)*vvis.variogram.lon_res)
                lat_lag.append((lat+1)*vvis.variogram.lat_res)
                time_lag.append((time+1)*vvis.variogram.t_res)

    df = pd.DataFrame({"lon_lag": lon_lag,
                       "lat_lag": lat_lag,
                       "t_lag": time_lag,
                       "u_semivariance": u_semivariance,
                       "v_semivariance": v_semivariance})
    df.to_csv(file_path, index=False)


def save_tuned_empirial_variogram_2d(vvis: VisualizeVariogram, view_range: List[int], file_path: str):
    """Convert hand-tuned variogram to dataframe and save."""

    space_lag, time_lag = [], []
    u_semivariance, v_semivariance = [], []
    for space in range(int(view_range[0]/vvis.variogram.res_tuple[0])):
        for time in range(int(view_range[1]/vvis.variogram.res_tuple[1])):
            u_semivariance.append(vvis.variogram.bins[space, time, 0])
            v_semivariance.append(vvis.variogram.bins[space, time, 1])
            space_lag.append((space+1)*vvis.variogram.space_res)
            time_lag.append((time+1)*vvis.variogram.t_res)

    df = pd.DataFrame({"space_lag": space_lag,
                       "t_lag": time_lag,
                       "u_semivariance": u_semivariance,
                       "v_semivariance": v_semivariance})
    df.to_csv(file_path, index=False)


def save_variogram_to_npy(variogram: Variogram, file_path: str):
    if variogram.bins is None:
        raise Exception("Need to build variogram first before you can save it!")

    data_to_save = {"bins": variogram.bins,
                    "bins_count": variogram.bins_count,
                    "res": variogram.res_tuple,
                    "units": variogram.units,
                    "detrend_metrics": variogram.bin_statistics
                    }
    np.save(file_path, data_to_save)
    print(f"\nSaved variogram data to: {file_path}")
