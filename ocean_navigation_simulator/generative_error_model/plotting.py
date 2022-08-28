"""
Various functions to aggregate buoy data into one object and interpolate forecasts/hindcasts
to spatio-temporal points for buoy data
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_buoy_data_time_collapsed(df: pd.DataFrame):
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    grid_lines = ax.gridlines(draw_labels=True, zorder=5)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black')

    for buoy_name in set(df["buoy"]):
        ax.scatter(df[df["buoy"] == buoy_name]["lon"], df[df["buoy"] == buoy_name]["lat"], marker=".")
    plt.show()


def plot_buoy_data_at_time_step(df: pd.DataFrame, plot: bool=False):
    fig = plt.figure(figsize=(10, 10))
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