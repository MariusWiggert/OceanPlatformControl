import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, List


def get_lag_all_pairs(df: pd.DataFrame, res_tuple: Tuple[int],\
    bins_tuple: Tuple[int], chunk_size:int) -> np.ndarray:
    n = df.shape[0]
    lon_res, lat_res, t_res = res_tuple
    lon_bins, lat_bins, t_bins = bins_tuple
    # get indices of pairs
    i,j = np.triu_indices(n, k=1)
    print(f" number of pairs: {len(i)}")

    # convert time axis to datetime
    df["time"] = pd.to_datetime(df["time"])

    # make bins
    bins = np.zeros((lon_bins,lat_bins,t_bins,2)) # 5 km, 5km, 6hrs steps
    bins_count = np.zeros((lon_bins, lat_bins, t_bins,2))

    # convert relevant to numpy before accessing
    time = df["time"].to_numpy()
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    u_error = df["u_error"].to_numpy()
    v_error = df["v_error"].to_numpy()

    offset = 0
    for chunk in tqdm(range(chunk_size, len(i), chunk_size)):
        # get values for chunk
        idx_i = i[0+offset:chunk]
        idx_j = j[0+offset:chunk]

        # get lags
        # TODO: find way to exclude pairs form same buoy
        t_lag = np.floor((np.absolute((time[idx_i] - time[idx_j])/3.6e12)/t_res).astype(float)).astype(int)
        lon_lag = np.floor(np.absolute(lon[idx_i] - lon[idx_j])/lon_res).astype(int)
        lat_lag = np.floor(np.absolute(lat[idx_i] - lat[idx_j])/lat_res).astype(int)
        u_squared_diff = np.square(u_error[idx_i] - u_error[idx_j])
        v_squared_diff = np.square(v_error[idx_i] - v_error[idx_j])
        squared_diff = np.array(np.vstack((u_squared_diff, v_squared_diff)).reshape((-1,2)))

        # assign values to bin
        if np.sum(bins) == 0:
            # from: https://stackoverflow.com/questions/51092737/vectorized-assignment-in-numpy
            # add to relevant bin
            np.add.at(bins, (lon_lag, lat_lag, t_lag), squared_diff)
            # add to bin count
            np.add.at(bins_count, (lon_lag, lat_lag, t_lag), [1,1])
        else:
            bins_temp = np.zeros((lon_bins,lat_bins,t_bins,2))
            bins_count_temp = np.zeros(bins_temp.shape)
            # perform operations
            np.add.at(bins_temp, (lon_lag, lat_lag, t_lag), squared_diff)
            np.add.at(bins_count_temp, (lon_lag, lat_lag, t_lag), [1,1])
            bins += bins_temp
            bins_count += bins_count_temp

        offset += chunk_size

    bins[lon_lag, lat_lag, t_lag] /= bins_count[lon_lag, lat_lag, t_lag]
    return bins, bins_count
    