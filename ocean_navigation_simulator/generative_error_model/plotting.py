"""
Various functions to aggregate buoy data into one object and interpolate forecasts/hindcasts
to spatio-temporal points for buoy data
"""

import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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
        return fig, scatter_plot


def interactive_sampled_noise(data: xr.Dataset):
    """Interactive viewer for generated current errors."""

    u_error = data["u_error"].values.transpose((1, 0, 2))
    v_error = data["v_error"].values.transpose((1, 0, 2))
    print(type(v_error))
    time = data["time"].values
    fig = plt.figure(figsize=(20, 12))
    plt.subplots_adjust(bottom=0.35)
    axtime = plt.axes([0.25, 0.2, 0.65, 0.03])
    slider_time = Slider(axtime, 'time', 0, len(time), valinit=0, valstep=1)

    ax = fig.add_subplot()
    img = ax.imshow(u_error[:, :, 0])
    fig.colorbar(mappable=img, ax=ax)

    def update(val):
        # amp is the current value of the slider
        time_idx = slider_time.val
        # update curve
        new_img = u_error[:, :, time_idx]
        img.set_data(new_img)
        img.set_clim([new_img.min(), new_img.max()])
        fig.canvas.draw()
        fig.canvas.flush_events()

    # call update function on slider value change
    slider_time.on_changed(update)
    plt.show()

