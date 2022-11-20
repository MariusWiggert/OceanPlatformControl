import numpy as np
from ocean_navigation_simulator.data_sources.DataSource import DataSource
import datetime
from typing import AnyStr, List, Optional, Union, Dict
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.environment.Arena import Arena


def calc_fmrc_errors(problem: Problem, arena: Arena, t_horizon_in_h: int,
                     deg_around_x0_xT_box: float, T_goal_in_seconds: int) -> Dict:
    """Calculating the forecast errors for a specific problem and returns a dict with RMSE, vector correlation, and abs angle difference.
    First we calculate the errors for each forecast for t_horizon_in_h and get the mean over space.
    Then we get the mean over all forecasts that the problem is affected by.
    The output arrays are then 1D arrays with the error across forecast_lag_time in hours (or standard unit of fc).
    """
    # Get intervals over which to calculate the error
    t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
        x_0=problem.start_state.to_spatio_temporal_point(),
        x_T=problem.end_region,
        deg_around_x0_xT_box=deg_around_x0_xT_box,
        temp_horizon_in_s=T_goal_in_seconds,
    )

    # Get the starting forecast time for the first forecast of the problem
    forecast_time_start = arena.ocean_field.forecast_data_source.check_for_most_recent_fmrc_dataframe(problem.start_state.date_time)
    t_horizon_in_h = int(t_horizon_in_h)

    # initialize lists for logging arrays for the individual forecasts
    RMSE_across_fmrc = []
    angle_diff_across_fmrc = []
    vec_corr_across_fmrc = []

    # Iterate over all forecasts until forecast_time_start is bigger than final time
    while forecast_time_start < t_interval[1]:
        # % get forecast for first time-interval
        fc_data = arena.ocean_field.forecast_data_source.get_data_over_area(
            x_interval=x_interval,
            y_interval=y_interval,
            t_interval=[forecast_time_start, forecast_time_start + datetime.timedelta(hours=t_horizon_in_h)])
        # % get hindcast data over a slightly bigger area!
        x_increment = 0.2
        y_increment = 0.2
        hc_data = arena.ocean_field.hindcast_data_source.get_data_over_area(
            x_interval=[x - ((-1) ** i) * x_increment for i, x in enumerate(x_interval)],
            y_interval=[y - ((-1) ** i) * y_increment for i, y in enumerate(y_interval)],
            t_interval=[forecast_time_start - datetime.timedelta(hours=3),
                        forecast_time_start + datetime.timedelta(hours=t_horizon_in_h) + datetime.timedelta(hours=3)])
        hc_correct_size = hc_data.interp_like(fc_data)
        # get data
        RMSE_across_fmrc.append(calc_speed_RMSE(fc_data['water_u'].data, fc_data['water_v'].data, hc_correct_size['water_u'].data,
                        hc_correct_size['water_v'].data))
        angle_diff_across_fmrc.append(calc_abs_angle_difference(fc_data['water_u'].data, fc_data['water_v'].data, hc_correct_size['water_u'].data,
                                  hc_correct_size['water_v'].data))
        vec_corr_across_fmrc.append(calc_vector_corr_over_time(fc_data.to_array().to_numpy().transpose((1, 3, 2, 0)),
                                   hc_correct_size.to_array().to_numpy().transpose((1, 3, 2, 0))))
        # prep for next round
        next_time = forecast_time_start + datetime.timedelta(days=1, hours=5)
        forecast_time_start = arena.ocean_field.forecast_data_source.check_for_most_recent_fmrc_dataframe(next_time)

    # Return the results dict
    return {
        "RMSE_velocity": np.array(RMSE_across_fmrc).mean(axis=0),
        "angle_diff": np.array(angle_diff_across_fmrc).mean(axis=0),
        "vector_correlation": np.array(vec_corr_across_fmrc).mean(axis=0),
    }


# HELPER FUNCTIONS
def calc_speed_RMSE(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast):
    """Helper function to calculate the RMSE on current speed."""
    RMSE_speed = np.sqrt(
        (u_data_forecast - u_data_hindcast) ** 2 + (v_data_forecast - v_data_hindcast) ** 2
    ).mean(axis=(1, 2))
    return RMSE_speed

def calc_abs_angle_difference(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast):
    return np.abs(
        np.arctan2(v_data_hindcast, u_data_hindcast) - np.arctan2(v_data_forecast, u_data_forecast)
    ).mean(axis=(1, 2))


def calc_vector_corr_over_time(forecast, hindcast, sigma_diag=0, remove_nans: bool = False):
    # run it over a for loop
    vec_corr_over_time = []
    for time_idx in range(forecast.shape[0]):
        vec_corr = calc_vector_correlation(
            forecast[time_idx, ...],
            hindcast[time_idx, ...],
            sigma_diag=sigma_diag,
            remove_nans=remove_nans,
        )
        vec_corr_over_time.append(vec_corr)
    # compile it to a np array
    return np.array(vec_corr_over_time)


def calc_vector_correlation(
    forecast: np.ndarray,
    hindcast: np.ndarray,
    print_out: bool = False,
    sigma_diag: float = 0,
    remove_nans: bool = False,
):
    # shape for forecast and hindcast: (lon, lat,2)
    # Flatten out the vectors
    forecast_vec = np.swapaxes(forecast, -1, 0).reshape((2, -1))
    hindcast_vec = np.swapaxes(hindcast, -1, 0).reshape((2, -1))
    if remove_nans:
        m = np.bitwise_or.reduce(
            np.logical_or(np.isnan(forecast_vec), np.isnan(hindcast_vec)), axis=0
        )
        forecast_vec = forecast_vec[:, ~m]
        hindcast_vec = hindcast_vec[:, ~m]

    # Step 1: calculate the correlation matrix
    full_variable_vec = np.vstack((forecast_vec, hindcast_vec))
    Covariance_matrix = np.cov(full_variable_vec)
    # calculation for vector correlation
    Sigma_11 = Covariance_matrix[:2, :2] + sigma_diag * np.eye(2)
    Sigma_22 = Covariance_matrix[2:, 2:] + sigma_diag * np.eye(2)
    Sigma_12 = Covariance_matrix[:2, 2:]
    Sigma_21 = Covariance_matrix[2:, :2]
    # Matrix multiplications
    vector_correlation = np.trace(
        np.linalg.inv(Sigma_11) @ Sigma_12 @ np.linalg.inv(Sigma_22) @ Sigma_21
    )
    if print_out:
        print("vector_correlation is : ", vector_correlation)
    return vector_correlation
