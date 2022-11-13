##
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

#
from ocean_navigation_simulator.ocean_observer.ExperimentRunner import ExperimentRunner
from ocean_navigation_simulator.ocean_observer.PredictionsAndGroundTruthOverArea import (
    PredictionsAndGroundTruthOverArea,
)

full_path = Path("ablation_study/configs_GP/gm_GP_025_12.yaml")
arena = ArenaFactory.create(
    scenario_name=full_path.resolve().stem, folder_scenario=full_path.parent
)
fc = arena.ocean_field.forecast_data_source
hc = (
    arena.ocean_field.hindcast_data_source
)  # .DataArray.interp_like(arena.ocean_field.forecast_data_source.DataArray)
j = 15
lat_interv = fc.DataArray["lat"][[j, -j]]
lon_interv = fc.DataArray["lon"][[j, -j]]
margin_space = 1 / 12
lat_interv_hc = (lat_interv[0] - margin_space, lat_interv[1] + margin_space)
lon_interv_hc = (lon_interv[0] - margin_space, lon_interv[1] + margin_space)

start = datetime.datetime(2022, 5, 3, 12, 30, tzinfo=datetime.timezone.utc) + datetime.timedelta(
    hours=1
)
delta = datetime.timedelta(hours=1)
num_lags = 36
delta_lags = datetime.timedelta(hours=num_lags)
margin = datetime.timedelta(hours=1)
res = xr.Dataset()
len_days = 90  # 4 * 30 + 11
list_forecast_hindcast = []
for i in range(len_days):
    if not i % 10:
        print(i + 1, "/", len_days)
    try:
        f = fc.get_data_over_area(lon_interv, lat_interv, [start, start + delta + delta_lags]).isel(
            time=slice(0, num_lags)
        )

        lat = f["lat"]
        lon = f["lon"]
        lat_2 = np.arange(20, 28 + margin_space, margin_space)
        lon_2 = np.arange(-96, -76 + margin_space, margin_space)
        # lat_2 = np.concatenate((np.arange(20, lat[0], 1 / 12), lat, np.arange(1 / 12 + lat[-1], 28, 1 / 12)))
        # lon_2 = np.concatenate((np.arange(-97, lon[0], 1 / 12), lon, np.arange(1 / 12 + lon[-1], -77, 1 / 12)))
        time_2 = f["time"]
        base_grid = xr.Dataset(
            data_vars=dict(
                water_u=(
                    ("lon", "lat", "time"),
                    np.full([len(lon_2), len(lat_2), len(time_2)], np.nan),
                ),
                water_v=(
                    ("lon", "for ", "time"),
                    np.full([len(lon_2), len(lat_2), len(time_2)], np.nan),
                ),
            ),
            coords=dict(
                lon=lon_2,
                lat=lat_2,
                time=time_2,
            ),
        )

        f = f.interp_like(base_grid)
        h = hc.get_data_over_area(
            lon_interv_hc, lat_interv_hc, [start - margin, start + delta_lags + margin]
        )
        h = h.assign_coords(depth=f.depth.to_numpy().tolist())
        h_interp = h.interp_like(f)
        if len(f.time) != num_lags or len(h_interp.time) != num_lags:
            raise ValueError
        error = f - h_interp
        res = error.combine_first(res)
        f_renamed = xr.merge(
            [
                f,
                f.rename_vars(water_u="initial_forecast_u", water_v="initial_forecast_v"),
            ]
        )
        obj = PredictionsAndGroundTruthOverArea(f_renamed, h)
        list_forecast_hindcast.append((obj.predictions_over_area, obj.ground_truth))
    except ValueError:
        print(f"error with day: {start}. Skipped!")
    start += delta
print("collected data")
# Format like ExperimentRunner.visualize_all_noise()
list_error = [
    pred[0][["initial_forecast_u", "initial_forecast_v"]].rename(
        {"initial_forecast_u": "water_u", "initial_forecast_v": "water_v"}
    )
    - pred[1]
    for pred in list_forecast_hindcast
]
list_forecast = [
    pred[0][["initial_forecast_u", "initial_forecast_v"]].rename(
        {"initial_forecast_u": "water_u", "initial_forecast_v": "water_v"}
    )
    for pred in list_forecast_hindcast
]
array_error_per_time = []
array_forecast_per_time = []
for t in range(len(list_error[0]["time"])):
    array_error_per_time.append(
        np.array(
            [
                list_day.isel(time=t)
                .assign(
                    magnitude=lambda x: (x.water_u**2 + x.water_v**2) ** 0.5,
                    angle=lambda x: np.arctan2(x.water_v, x.water_u),
                )
                .to_array()
                .to_numpy()
                for list_day in list_error
            ]
        )
    )
    array_forecast_per_time.append(
        np.array(
            [
                list_day.isel(time=t)
                .assign(magnitude=lambda x: (x.water_u**2 + x.water_v**2) ** 0.5)
                .to_array()
                .to_numpy()
                for list_day in list_forecast
            ]
        )
    )
array_error_per_time = np.array(array_error_per_time)
array_forecast_per_time = np.array(array_forecast_per_time)

print("plotting")
# -----------------------------------------
# QQ plots
# -----------------------------------------

# qq-plot on the magnitude
n_dims = 2  # array_per_time.shape[2]
max_lags_to_plot = 48
# Only consider the first 24 predictions
factor = 3
max_lags_to_plot_grid = 12
n_col = min(len(array_error_per_time), max_lags_to_plot_grid) // factor
n_row = factor
array_error_per_time = array_error_per_time[: n_col * n_row]
array_forecast_per_time = array_forecast_per_time[: n_col * n_row]
dim_name = ["longitude", "latitude", "magnitude"]
for dim_current in range(n_dims):
    fig, ax = plt.subplots(n_row, n_col)  # , sharey=True, sharex=True)
    fig.suptitle(f"QQ-plots for each lag, {dim_name[dim_current]}")
    for i in range(len(array_error_per_time)):
        if True:
            x = ExperimentRunner._remove_nan_and_flatten(array_error_per_time[i][:, dim_current])
            ax_now = ax[i // n_col, i % n_col]
            sm.qqplot(x, line="s", ax=ax_now)  # , fit=True)
            ax_now.set_title(f"lag:{i}")
            ax_now.set_ylim(-2.5, 2.5)
    # plt.legend(f'current: {["u", "v", "magn"][dim_current]}')
    plt.subplots_adjust(hspace=0.348, wspace=0.229)
    plt.show()

# Plot the general qq plots
dims_first = np.moveaxis(array_error_per_time, 2, 0)
values_flattened = dims_first.reshape(len(dims_first), -1)
df_describe = pd.DataFrame(values_flattened.T, columns=["u", "v", "magn", "angle"]).dropna()
fig, ax = plt.subplots(2, 1)
# sm.qqplot(ExperimentRunner._remove_nan_and_flatten(dims_first[0]), line='s', ax=ax[0])
# sm.qqplot(ExperimentRunner._remove_nan_and_flatten(dims_first[1]), line='s', ax=ax[1])
# [a.set_ylim(-2.5, 2.5) for a in ax]
# # ax[0].set_xlabel("TheoriticalDimension: u")
# # ax[0].legend(f'current: {["u", "v", "magn"][dim_current]}')
# # ax[1].set_xlabel("Dimension: v")
# ax[0].set_title("Dimension: longitude")
# ax[1].set_title("Dimension: latitude")
# plt.subplots_adjust(hspace=.3)


# -----------------------------------------
# Error wrt forecast magnitude
# -----------------------------------------
print("Error wrt forecast magnitude")
dims_to_plot = ["u", "v", "magnitude"]
# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
# fig.suptitle(f'Scatter plot of the error')
# for i, dim in enumerate(dims_to_plot):
#     ax[i].scatter(array_forecast_per_time[:, :, i].flatten(),
#                   array_error_per_time[:, :, i].flatten(), s=0.04)
#     # ax[i].set_title(f"Dimension:{dim}")
# m = max(-np.nanmin(array_forecast_per_time), np.nanmax(array_forecast_per_time))
# plt.xlim(-m, m)
# plt.ylim(np.nanmin(array_error_per_time), np.nanmax(array_error_per_time))
# plt.xlabel("forecast")
# plt.ylabel("error")

print("Error wrt forecast magnitude same aspect")
dims_to_plot = ["u", "v", "magnitude"]
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
fig.suptitle("Error scatter plot")
for i, dim in enumerate(dims_to_plot):
    x, y = (
        array_forecast_per_time[:, :, i].flatten(),
        array_error_per_time[:, :, i].flatten(),
    )
    nans = np.isnan(x + y)
    x, y = x[~nans], y[~nans]
    local_ax = ax[i]
    local_ax.scatter(x, y, s=0.04, alpha=0.009)

    # Print the regression line
    b, a = np.polyfit(x, y, deg=1)
    xseq = np.linspace(np.nanmin(x), np.nanmax(x), num=1000)
    local_ax.plot(xseq, a + b * xseq, color="k", lw=2.5)

    # r-squared
    p = np.poly1d((b, a))
    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    r2 = ssreg / sstot

    local_ax.set_title(f"{dim_name[i]}, slope: {b:.3f}, r2: {r2:.3f}")
    # local_ax.set_aspect("equal")
    local_ax.set_xlabel("forecast speed [m/s]")
    if i == 2:
        local_ax.set_ylabel("error [m/s]")
margin = 0.12
m = max(-np.nanmin(array_forecast_per_time), np.nanmax(array_forecast_per_time)) + margin
plt.xlim(-m, m)
# plt.ylim(np.nanmin(array_error_per_time), np.nanmax(array_error_per_time))
plt.ylim(-2, 2)

# -----------------------------------------
# Describe stats
# -----------------------------------------
print("\n\n\n\nDETAIL about the whole dataset\n", df_describe.describe())

# -----------------------------------------
# Histogram
# -----------------------------------------
# nbins = 250
# fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
# df_describe.hist("u", ax=ax[0], bins=nbins, density=True)
#
# ExperimentRunner._add_normal_on_plot(df_describe["u"], ax[0], "longitude")
# df_describe.hist("v", ax=ax[1], bins=nbins, density=True)
# ExperimentRunner._add_normal_on_plot(df_describe["v"], ax[1], "latitude")
# ax[0].set_xlim(-1, 1)
# ax[0].legend()
# ax[1].legend()

#
dims_to_plot_ci = ["u", "v", "magn"]
fig, ax = plt.subplots(len(dims_to_plot_ci), 1, sharex=True, sharey=False)

# for each lag:
stat_lags = []
for i in range(len(array_error_per_time)):
    stat_lags.append(
        pd.DataFrame(
            np.array([abs(e[i]).flatten() for e in dims_first]).T,
            columns=["u", "v", "magn", "angle"],
        ).describe()
    )
for i, axi in enumerate(ax):
    dim = dims_to_plot_ci[i]
    mean = np.array([s[dim]["mean"] for s in stat_lags])
    x = list(range(len(mean)))
    # 95% CI
    ci = np.array([s[dim]["std"] * 1.96 / np.sqrt(s[dim]["count"]) for s in stat_lags])
    axi.plot(mean, color="black", lw=0.7)
    axi.fill_between(x, mean - ci, mean + ci, color="blue", alpha=0.1)
    axi.set_title(f"{dim_name[i]}")
plt.xlabel("Lag [h]")
plt.ylabel("Absolute error [m/s]")
fig.suptitle("Evolution of the absolute error with respect to the lag")
plt.xticks(np.arange(0, len(mean), 3))

plt.plot()

print("over :)")
