import datetime
from typing import Dict

import matplotlib.pyplot as plt

from ocean_navigation_simulator.utils.units import format_datetime_x_axis

# For each metric we have a plotting function


def plot_r2(metrics_dict: Dict, ax: plt.axes, X_LABELSIZE=10):
    time_axis = [
        datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)
        for time in metrics_dict["time"]
    ]
    format_datetime_x_axis(ax)
    ax.plot(time_axis, metrics_dict["r2"], label="r2")
    ax.axhline(y=0, linestyle="--", color="black")
    ax.set_title("r2 score")
    ax.tick_params(axis="x", labelsize=X_LABELSIZE)


def plot_rmse(metrics_dict: Dict, ax: plt.axes, X_LABELSIZE=10):
    time_axis = [
        datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)
        for time in metrics_dict["time"]
    ]
    format_datetime_x_axis(ax)
    ax.plot(time_axis, metrics_dict["rmse_improved"], label="improved", linewidth=0.5)
    ax.plot(
        time_axis,
        metrics_dict["rmse_initial"],
        label="initial",
        linewidth=0.5,
        linestyle="--",
        color="black",
    )
    ax.legend()
    ax.axhline(y=0, linestyle="--", color="black")
    ax.set_title("RMSE")
    ax.tick_params(axis="x", labelsize=X_LABELSIZE)


def plot_vector_correlation(metrics_dict: Dict, ax: plt.axes, X_LABELSIZE=10):
    # Plot Vector Correlation of improved forecast
    time_axis = [
        datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)
        for time in metrics_dict["time"]
    ]
    format_datetime_x_axis(ax)
    ax.plot(time_axis, metrics_dict["vector_correlation_improved"], label="improved", linewidth=0.5)
    ax.plot(
        time_axis,
        metrics_dict["vector_correlation_initial"],
        label="initial",
        linewidth=0.5,
        linestyle="--",
        color="black",
    )
    ax.set_title("Vector Correlation")
    ax.legend()
    ax.tick_params(axis="x", labelsize=X_LABELSIZE)


def plot_vector_correlation_ratio(metrics_dict: Dict, ax: plt.axes, X_LABELSIZE=10):
    time_axis = [
        datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)
        for time in metrics_dict["time"]
    ]
    format_datetime_x_axis(ax)
    ax.set_title("vector correlation ratio")
    ax.plot(time_axis, metrics_dict["vector_correlation_ratio"], linewidth=0.5)
    ax.axhline(y=1, linestyle="--", color="black")
    ax.tick_params(axis="x", labelsize=X_LABELSIZE)
