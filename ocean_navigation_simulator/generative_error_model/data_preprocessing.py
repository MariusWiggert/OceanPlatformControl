"""
Various functions to aggregate buoy data into one object and interpolate forecasts/hindcasts
to spatio-temporal points for buoy data
"""

from ocean_navigation_simulator.environment.PlatformState import PlatformState
from ocean_navigation_simulator.environment.data_sources import OceanCurrentField
from ocean_navigation_simulator.utils import units

import numpy as np
import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

# TODO: ensure space-time range is adequate for interpolation (PlatformState)

def plot_buoy_data_time_collapsed(df: pd.DataFrame):
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    grid_lines = ax.gridlines(draw_labels=True, zorder=5)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black')

    for buoy_name in set(df["buoy"]):
        ax.scatter(df[df["buoy"] == buoy_name]["lon"], df[df["buoy"] == buoy_name]["lat"], marker=".")
    plt.show()

def plot_buoy_data_at_time_step(df: pd.DataFrame, plot: bool=False):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    grid_lines = ax.gridlines(draw_labels=True, zorder=5)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    ax.add_feature(cfeature.LAND, zorder=3, edgecolor="black")
    time = list(set(df["time_offset"]))[0]
    scatter_plot, = ax.plot(df[df["time_offset"] == time]["lon"], df[df["time_offset"] == time]["lat"], marker="o", ls="")
    if plot:
        plt.show()
    else:
        return fig, scatter_plot, ax

def interp_xarray(df: pd.DataFrame, ocean_field: OceanCurrentField, data_source: str, n: int=10) -> pd.DataFrame: 
    """
    Interpolates the hindcast data to spatio-temporal points of buoy measurements

    data_source: str {hindcast_data_source, forecast_data_source}
    """

    df[f"u_{data_source.split('_')[0]}"] = 0
    df[f"v_{data_source.split('_')[0]}"] = 0
    # convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])

    for i in tqdm(range(0, df.shape[0], n)):
        hindcast_interp = getattr(ocean_field, data_source).DataArray.interp(time=df.iloc[i:i+n]["time"],
                                                                            lon=df.iloc[i:i+n]["lon"],
                                                                            lat=df.iloc[i:i+n]["lat"])
        # add columns to dataframe
        df[f"u_{data_source.split('_')[0]}"].iloc[i:i+n] = hindcast_interp["water_u"].values.diagonal().diagonal()
        df[f"v_{data_source.split('_')[0]}"].iloc[i:i+n] = hindcast_interp["water_v"].values.diagonal().diagonal()
    return df

def interp_hincast_casadi(df: pd.DataFrame, hindcast_x_interval: List[float], hindcast_y_interval: List[float],
                        hindcast_date_time: np.datetime64, ocean_field: OceanCurrentField) -> pd.DataFrame:
    """
    Interpolates the hindcast data to spatio-temporal points of buoy measurements.
    """

    platform_state = PlatformState(lon=units.Distance(deg=np.mean(hindcast_x_interval)),
                                    lat=units.Distance(deg=np.mean(hindcast_y_interval)),
                                    date_time=hindcast_date_time)                     
    ocean_field.hindcast_data_source.update_casadi_dynamics(state=platform_state)

    # convert buoy time axis to posix
    time_posix = units.get_posix_time_from_np64(df["time"])
    spatio_temporal_points = np.array([time_posix, df["lat"], df["lon"]]).T

    interp_u = []
    interp_v = []
    for i in tqdm(range(spatio_temporal_points.shape[0])):
        interp_u.append(float(ocean_field.hindcast_data_source.u_curr_func(spatio_temporal_points[i])))
        interp_v.append(float(ocean_field.hindcast_data_source.v_curr_func(spatio_temporal_points[i])))
    
    df["u_hind"] = interp_u
    df["v_hind"] = interp_v
    return df