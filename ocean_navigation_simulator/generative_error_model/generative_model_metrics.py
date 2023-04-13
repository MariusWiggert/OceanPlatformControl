from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


def get_metrics() -> Dict[str, Callable[[pd.DataFrame, pd.DataFrame], Dict[str, Any]]]:
    """Returns a dict where each key-value pairs corresponds to the metric name and its corresponding
    function."""

    return {"rmse": rmse, "vector_correlation": vector_correlation}


def rmse(ground_truth, predictions):
    return np.sqrt(
        ((ground_truth[0] - predictions[0]) ** 2 + (ground_truth[1] - predictions[1]) ** 2).mean()
    )


def rmse_pd(ground_truth: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, float]:
    rmse_val = np.sqrt(
        (
            (
                (ground_truth["u_error"] - synthetic_data["u_error"]) ** 2
                + (ground_truth["v_error"] - synthetic_data["v_error"]) ** 2
            )
        ).mean()
    )
    return {"rmse": rmse_val}


def rmse_over_time(df: pd.DataFrame, variables: Tuple[str, str]) -> Dict[str, List[float]]:
    df["hour"] = df["time"].apply(lambda x: x[:13])
    hours = sorted(set(df["hour"].tolist()))
    rmse_data = []
    for hour in hours:
        rmse_data.append(
            np.sqrt(
                (
                    (df[df["hour"] == hour][variables[0]]) ** 2
                    + (df[df["hour"] == hour][variables[1]]) ** 2
                ).mean()
            )
        )
    return {"rmse": rmse_data}


def rmse_over_time_xr(error: xr.Dataset, variables: Tuple[str, str]):
    rmse_val = np.sqrt(
        ((error[variables[0]] ** 2).mean(dim=("lon", "lat"))).values
        + ((error[variables[1]] ** 2).mean(dim=("lon", "lat"))).values
    )
    return rmse_val


def vector_correlation_over_time(data: pd.DataFrame) -> np.ndarray:
    data["hour"] = data["time"].apply(lambda x: x[:13])
    hours = sorted(set(data["hour"].tolist()))
    vec_corr = []
    for hour in hours:
        buoy_u = data[data["hour"] == hour]["u"]
        buoy_v = data[data["hour"] == hour]["v"]
        forecast_u = data[data["hour"] == hour]["u_forecast"]
        forecast_v = data[data["hour"] == hour]["v_forecast"]
        vec_corr.append(vector_correlation(buoy_u, buoy_v, forecast_u, forecast_v))
    return np.array(vec_corr)


def vector_correlation_over_time_xr(error: xr.Dataset, forecast: xr.Dataset) -> np.ndarray:
    # rename forecast variables and slice forecast to size of error
    renaming_map = {"longitude": "lon", "latitude": "lat", "utotal": "water_u", "vtotal": "water_v"}
    if any(item in forecast.dims for item in renaming_map.keys()):
        forecast = forecast.rename(renaming_map)

    # need to slice forecast to match size of error
    lon_range = [error["lon"].values.min(), error["lon"].values.max()]
    lat_range = [error["lat"].values.min(), error["lat"].values.max()]
    time_range = [error["time"].values.min(), error["time"].values.max()]
    forecast = forecast.sel(lon=slice(*lon_range), lat=slice(*lat_range), time=slice(*time_range))

    # convert error to float32
    for data_source in [forecast, error]:
        data_source["lon"] = data_source["lon"].astype(np.float32)
        data_source["lat"] = data_source["lat"].astype(np.float32)
        data_source["water_u"] = data_source["water_u"].astype(np.float32)
        data_source["water_v"] = data_source["water_v"].astype(np.float32)
    # Note: addition isnot cummutative in this case as axis will be different order
    # and thus a slice at a specific time would be differently ordered.
    ground_truth = forecast + error

    # compute vector correlation for each hour
    vector_correlation_per_hour = []
    for time in ground_truth["time"]:
        gt_u_error = ground_truth["water_u"].sel(time=time).values
        gt_v_error = ground_truth["water_v"].sel(time=time).values
        fc_u_error = forecast["water_u"].sel(time=time).values
        fc_v_error = forecast["water_v"].sel(time=time).values
        vector_correlation_per_hour.append(
            vector_correlation(gt_u_error, gt_v_error, fc_u_error, fc_v_error)
        )
    return np.array(vector_correlation_per_hour)


def vector_correlation(
    u_data_hindcast, v_data_hindcast, u_data_measured, v_data_measured, print_out=False
):
    """
    Calculates the vector correlation for two sets of measurements.
    """

    # Flatten out the vectors
    # first Stack them
    hindcast = np.stack((u_data_hindcast, v_data_hindcast))
    measured = np.stack((u_data_measured, v_data_measured))
    hindcast_vec = hindcast.reshape(2, -1)
    measured_vec = measured.reshape(2, -1)
    # Step 1: calculate the correlation matrix
    full_variable_vec = np.vstack((measured_vec, hindcast_vec))
    Covariance_matrix = np.cov(full_variable_vec)
    # calculation for vector correlation
    sigma_11 = Covariance_matrix[:2, :2]
    sigma_22 = Covariance_matrix[2:, 2:]
    sigma_12 = Covariance_matrix[:2, 2:]
    sigma_21 = Covariance_matrix[2:, :2]
    # Matrix multiplications
    epsilon = 0
    try:
        vector_correlation = np.trace(
            np.linalg.inv(sigma_11) @ sigma_12 @ np.linalg.inv(sigma_22) @ sigma_21
        )
    except Exception:
        vector_correlation = np.NaN
        while np.isnan(vector_correlation):
            epsilon += 5e-5
            vector_correlation = np.trace(
                np.linalg.inv(sigma_11 + epsilon * np.eye(2))
                @ sigma_12
                @ np.linalg.inv(sigma_22 + epsilon * np.eye(2))
                @ sigma_21
            )
            if epsilon > 1e-1:
                print("Could not compute vector correlation!")
                break
    return vector_correlation


def temporal_smoothness(data: np.ndarray) -> float:
    """Assumes axis=0 is time and the other two are space."""

    for i in range(data.shape[0] - 1):
        temp = np.abs(data[i + 1] - data[i]).squeeze()
        temp[np.where(np.isnan(temp))] = 0
        if i == 0:
            temp_smoothness = temp
        else:
            temp_smoothness += temp

    return temp_smoothness.sum() / ((data.shape[0] - 1) * np.prod(temp_smoothness.shape))
    # return (temp_smoothness / (data.shape[0] - 1)).mean()


def get_correlation(var1: np.ndarray, var2: np.ndarray, *args):
    data = np.vstack([var1, var2])
    data = np.array([np.array(var1).flatten(), np.array(var2).flatten(), *args])
    return np.corrcoef(data)


def plot_correlation_grid(df: pd.DataFrame):
    sns.heatmap(df.corr(), cbar=False, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1)
