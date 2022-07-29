import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import List, Dict
from tqdm import tqdm
import math


def get_variogram_value(ds: pd.DataFrame, lag: float) -> float:
    sum = 0
    max_lag = ds["Depth"].iloc[-1] - 0.25
    for i in np.arange(0.25, max_lag - lag, 0.25):
        val1 = ds[ds["Depth"] == i]["Nporosity"].values[0]
        val2 = ds[ds["Depth"] == (i+lag)]["Nporosity"].values[0]
        sum += (val1 - val2)**2
    return sum/((float(max_lag) - lag)*8)


def _compute_pair_entries(df: pd.DataFrame, dictionary: Dict):
    print(f"size of DataFrame: {df}")

    first_row_first = False
    if df["time"].iloc[0] < df["time"].iloc[1]:
        first_row_first = True

    if first_row_first:
        dictionary["time_lag"].append(df["time"].iloc[0] - df["time"].iloc[1])
        dictionary["lon_lag"].append(df["lon"].iloc[0] - df["lon"].iloc[1])
        dictionary["lat_lag"].append(df["lat"].iloc[0] - df["lat"].iloc[1])
        dictionary["u_error"].apend(df["u_error"].iloc[0] - df["u_error"].iloc[1])
        dictionary["v_error"].append(df["v_error"].iloc[0] - df["u_error"].iloc[1])
    else:
        dictionary["time_lag"].append(df["time"].iloc[1] - df["time"].iloc[0])
        dictionary["lon_lag"].append(df["lon"].iloc[1] - df["lon"].iloc[0])
        dictionary["lat_lag"].append(df["lat"].iloc[1] - df["lat"].iloc[0])
        dictionary["u_error"].append(df["u_error"].iloc[1] - df["u_error"].iloc[0])
        dictionary["v_error"].append(df["v_error"].iloc[1] - df["u_error"].iloc[0])

    return dictionary


def select_cross_buoy_points_at_time(df: pd.DataFrame) -> pd.DataFrame:
    # get all pairs of points at specific time
    new_df_dict = {"time_lag":[], "lon_lag":[], "lat_lag":[], "u_error":[],\
                "v_error":[]}

    for time in tqdm(set(df["time"].tolist())):
        entries_at_time = df[df["time"] == time]
        # need time in datetime to get timedelta for lag
        entries_at_time["time"] = pd.to_datetime(df["time"])
        combination = list(itertools.combinations(entries_at_time["buoy"], 2))
        new_df_dict = [_compute_pair_entries(entries_at_time[entries_at_time["buoy"].isin(c)], new_df_dict) for c in combination]

    df_lag = pd.DataFrame(new_df_dict)
    return df_lag

def select_cross_buoy_points(df: pd.DataFrame) -> pd.DataFrame:
    # split all buoy trajectories into separate list
    # for loop over all lists
    # in each loop use list comprehension to get pair

    new_df_dict = {"time_lag":[], "lon_lag":[], "lat_lag":[], "u_error":[],\
                "v_error":[]}
    df["time"] = pd.to_datetime(df["time"])


    # get all pairs of data points except pairs between points of the same buoy


    buoy_names = df["buoy"]

    for time in tqdm(set(df["time"].tolist())):
        entries_at_time = df[df["time"] == time]
        # need time in datetime to get timedelta for lag
        entries_at_time["time"] = pd.to_datetime(df["time"])
        combination = list(itertools.combinations(entries_at_time["buoy"], 2))
        new_df_dict = [_compute_pair_entries(entries_at_time[entries_at_time["buoy"].isin(c)], new_df_dict) for c in combination]

    df_lag = pd.DataFrame(new_df_dict)
    return df_lag

def get_lag_all_pairs(df: pd.DataFrame) -> np.ndarray:
    n = df.shape[0]
    # get indices of pairs
    i,j = np.triu_indices(n, k=1)
    print(f" number of pairs: {len(i)}")

    # convert time axis to datetime
    df["time"] = pd.to_datetime(df["time"])

    # make bins
    t_res = 1
    lon_res = 5
    lat_res = 5
    bins = np.zeros((1000,1000,80,2)) # 5 km, 5km, 6hrs steps

    # calculate lags and diff squared on the fly
    chunk_size = 100000
    offset = 0

    # convert relevant to numpy before accessing
    time = df["time"].to_numpy()
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    u_error = df["u_error"].to_numpy()
    v_error = df["v_error"].to_numpy()

    for chunk in tqdm(range(chunk_size, len(i), chunk_size)):
        # get values for chunk
        idx_i = i[0+offset:chunk]
        idx_j = j[0+offset:chunk]

        print(f"range: {0+offset}:{chunk}")

        # get lags
        # TODO: find way to exclude pairs form same buoy
        t_lag = np.floor((np.absolute((time[idx_i] - time[idx_j])/3.6e12)/t_res).astype(float)).astype(int)
        lon_lag = np.floor(np.absolute(lon[idx_i] - lon[idx_j])/lon_res).astype(int)
        lat_lag = np.floor(np.absolute(lat[idx_i] - lat[idx_j])/lat_res).astype(int)
        u_squared_diff = np.square(u_error[idx_i] - u_error[idx_j])
        v_squared_diff = np.square(v_error[idx_i] - v_error[idx_j])
        squared_diff = np.array(np.vstack((u_squared_diff, v_squared_diff)).reshape((-1,2)))

        # assign values to bin
        print(lon_lag[:10], lat_lag[:10])
        print(len(i))
        print(sum(squared_diff))

        if np.sum(bins) == 0:
            bins[t_lag, lon_lag, lat_lag] = squared_diff
            bins = np.sum(bins, axis=3)/(2*len(idx_i))
        else:
            bins_empty = np.zeros((1000,1000,80,2))
            bins_empty[t_lag, lon_lag, lat_lag] = squared_diff
            bins_empty = np.sum(bins_empty, axis=3)/(2*len(idx_i))
            bins += bins

        offset += chunk_size
    return bins

    

if __name__ == "__main__":
    ds = pd.read_csv("test_data.csv")
    # has a gaussian distribution with (0,1)

    lags = ds["Depth"].tolist()[:-2]

    vals = []
    for lag in lags:
        vals.append(get_variogram_value(ds, lag))


    fig = plt.figure(figsize=(6,6))
    plt.scatter(lags, vals)
    plt.grid(visible=True, which='both')
    plt.show()