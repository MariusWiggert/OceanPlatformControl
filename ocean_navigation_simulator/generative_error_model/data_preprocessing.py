"""
Various functions to aggregate buoy data into one object and interpolate forecasts/hindcasts
to spatio-temporal points for buoy data
"""

from ocean_navigation_simulator.environment.PlatformState import PlatformState
from ocean_navigation_simulator.environment.data_sources import OceanCurrentField
from ocean_navigation_simulator.utils import units
from DrifterData import DrifterData

import numpy as np
import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# TODO: to create pd.DataFrame need information from DrifterData class
# to interpolate also need objects from the OceanCurrentField (for the hindcasts)
# how to make this easier to deal with.
# TODO: ensure space-time range is adequate for interpolation (PlatformState)

def aggregate_buoy_files(file_list: List[str], config: Dict) -> pd.DataFrame:
    """
    reads in all .nc files in the file_list
    filters data for desired time and space interval 
    creates a pandas.DataFrame
    """
    column_names = ["time", "lon", "lat", "u", "v", "buoy"]
    df = pd.DataFrame(columns = column_names)

    for file_path in file_list:
        ds = DrifterData.readNCFile(file_path)

        # select specific data
        time = ds["TIME"].values
        lon = ds["LONGITUDE"].values
        lat = ds["LATITUDE"].values
        u = ds["NSCT"].isel(DEPTH=-1).values
        v = ds["EWCT"].isel(DEPTH=-1).values
        buoy = [file_path.split("/")[-1].split(".")[0] for i in range(len(time))]
        df_temp = pd.DataFrame({"time":time, "lon":lon, "lat":lat, "u":u, "v":v, "buoy":buoy})

        # change time column to datetime
        df["time"] = pd.to_datetime(df["time"])

        # filtering conditions TODO: fix issue with -ve values for lon
        targeted_bbox = config["targeted_bbox"]
        targeted_time_range = config["targeted_time_range"]
        lon_cond = ((df_temp["lon"] <= targeted_bbox[0]) & (df_temp["lon"] >= targeted_bbox[2]))
        lat_cond = ((df_temp["lat"] >= targeted_bbox[1]) & (df_temp["lat"] <= targeted_bbox[3]))
        time_cond = ((df_temp["time"] >= np.datetime64(targeted_time_range[0])) & (df_temp["time"] <= np.datetime64(targeted_time_range[1])))

        # filtering and concat to df
        df_temp = df_temp.loc[(lon_cond & lat_cond & time_cond)]
        df = pd.concat([df, df_temp])

    return df

def plot_buoy_data(df: pd.DataFrame):
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    grid_lines = ax.gridlines(draw_labels=True, zorder=5)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black')

    for buoy_name in set(df["buoy"]):
        ax.scatter(df[df["buoy"] == buoy_name]["lon"], df[df["buoy"] == buoy_name]["lat"], marker=".")
    plt.show()

def interp_hindcast_xarray(df: pd.DataFrame, ocean_field: OceanCurrentField, n: int=10) -> pd.DataFrame: 
    """
    Interpolates the hindcast data to spatio-temporal points of buoy measurements
    """

    from tqdm import tqdm
    df["u_hind"] = 0
    df["v_hind"] = 0
    # convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])

    for i in tqdm(range(0, df.shape[0], n)):
        # hindcast_interp = ds_hind.interp(time=df.iloc[i:i+n]["time"],
        #                                 lon=df.iloc[i:i+n]["lon"],
        #                                 lat=df.iloc[i:i+n]["lat"])
        hindcast_interp = ocean_field.hindcast_data_source.DataArray.interp(time=df.iloc[i:i+n]["time"],
                                                                            lon=df.iloc[i:i+n]["lon"],
                                                                            lat=df.iloc[i:i+n]["lat"])
        # add columns to dataframe
        df["u_hind"][i:i+n] = hindcast_interp["water_u"].values.diagonal().diagonal()
        df["v_hind"][i:i+n] = hindcast_interp["water_v"].values.diagonal().diagonal()
    return df

def interp_hincast_casadi(hindcast_x_interval: float, hindcast_y_interval: float, hincast_date_time: np.datetime64,
                        buoy_time: List[float], buoy_lat: List[float], buoy_lon: List[float]) -> List[float]:
    """
    Interpolates the hindcast data to spatio-temporal points of buoy measurements
    """

    platform_state = PlatformState(lon=units.Distance(deg=np.mean(hincast_x_interval)),
                                    lat=units.Distance(deg=np.mean(hindcast_y_interval)),
                                    date_time=hincast_date_time)                     
    ocean_field.hindcast_data_source.update_casadi_dynamics(state=platform_state)

    # convert buoy time axis to posix
    time_posix = units.get_posix_time_from_np64(buoy_time)
    spatio_temporal_points = np.array([time_posix, buoy_lat, buoy_lon]).T

    interp_u = []
    interp_v = []
    for i in range(spatio_temporal_points.shape[0]):
        interp_u.append(ocean_field.hindcast_data_source.u_curr_func(spatio_temporal_points[i]))
        interp_v.append(ocean_field.hindcast_data_source.v_curr_func(spatio_temporal_points[i]))
    return interp_u, interp_v