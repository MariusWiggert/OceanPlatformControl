import numpy as np
import xarray as xr
import pandas as pd
from typing import Callable, Dict, Any, List


def get_metrics() -> Dict[str, Callable[[pd.DataFrame, pd.DataFrame], Dict[str, Any]]]:
    """Returns a dict where each key-value pairs corresponds to the metric name and its corresponding
    function."""

    return {"rmse": rmse, "vector_correlation": vector_correlation}


def rmse(ground_truth: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, float]:
    rmse_val = np.sqrt(((ground_truth["u_error"]-synthetic_data["u_error"])**2 +
                        (ground_truth["v_error"]-synthetic_data["v_error"])**2)).mean()
    return {"rmse": rmse_val}


def rmse_over_time(df: pd.DataFrame) -> Dict[str, List[float]]:
    df["hour"] = df["time"].apply(lambda x: x[:13])
    hours = sorted(set(df["hour"].tolist()))
    rmse_data = []
    for hour in hours:
        rmse_data.append(np.sqrt(((df[df["hour"] == hour]["u_error"])**2 +
                                  (df[df["hour"] == hour]["v_error"])**2)).mean())
    return {"rmse": rmse_data}


def rmse_over_time_xr(error: xr.Dataset):
    rmse_val = np.sqrt(((error["u_error"]**2).mean(dim=("lon", "lat"))).values +
                       ((error["v_error"]**2).mean(dim=("lon", "lat"))).values)
    return rmse_val


def vector_correlation_over_time_xr(error: xr.Dataset, forecast: xr.Dataset):
    # rename forecast variables and slice forecast to size of error
    forecast = forecast.rename({"longitude": "lon",
                                "latitude": "lat",
                                "utotal": "u_error",
                                "vtotal": "v_error"})

    # need to slice forecast to match size of error
    lon_range = [error["lon"].values.min(), error["lon"].values.max()]
    lat_range = [error["lat"].values.min(), error["lat"].values.max()]
    time_range = [error["time"].values.min(), error["time"].values.max()]
    forecast = forecast.sel(lon=slice(*lon_range),
                            lat=slice(*lat_range),
                            time=slice(*time_range))

    # convert error to float32
    error["lon"] = error["lon"].astype(np.float32)
    error["lat"] = error["lat"].astype(np.float32)
    error["u_error"] = error["u_error"].astype(np.float32)
    error["v_error"] = error["v_error"].astype(np.float32)
    ground_truth = error + forecast

    # compute vector correlation for each hour
    vector_correlation_per_hour = []
    for time in ground_truth["time"]:
        gt_u_error = ground_truth["u_error"].sel(time=time).values
        gt_v_error = ground_truth["v_error"].sel(time=time).values
        fc_u_error = forecast["u_error"].sel(time=time).values
        fc_v_error = forecast["v_error"].sel(time=time).values
        vector_correlation_per_hour.append(vector_correlation(gt_u_error, gt_v_error, fc_u_error, fc_v_error))
    return np.array(vector_correlation_per_hour)


def vector_correlation(u_data_hindcast, v_data_hindcast, u_data_measured, v_data_measured, print_out=False):
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
        vector_correlation = np.trace(np.linalg.inv(sigma_11) @ sigma_12 @ np.linalg.inv(sigma_22) @ sigma_21)
    except:
        vector_correlation = np.NaN
        while np.isnan(vector_correlation):
            epsilon += 5e-5
            vector_correlation = np.trace(np.linalg.inv(sigma_11 + epsilon*np.eye(2)) @ sigma_12 @ np.linalg.inv(sigma_22 + epsilon*np.eye(2)) @ sigma_21)
            if epsilon > 1e-1:
                print("Could not compute vector correlation!")
                break
    return vector_correlation
