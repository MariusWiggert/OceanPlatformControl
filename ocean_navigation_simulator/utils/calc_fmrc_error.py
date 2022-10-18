from datetime import timedelta

import numpy as np
import xarray as xr

import ocean_navigation_simulator.utils as utils


# Note: for now just the logic for HYCOM
def calc_fmrc_errors(problem, T_horizon, deg_around_x0_xT_box, hours_to_abs_time=3600):
    # Step 0: make sure we're in analytical current case
    if problem.hindcast_data_source["data_source_type"] == "analytical_function":
        raise ValueError(
            "problem.hindcast_data_source['data_source_type'] is 'analytical_function'"
        )
    if problem.forecast_data_source["data_source_type"] == "analytical_function":
        raise ValueError(
            "problem.forecast_data_source['data_source_type'] is 'analytical_function'"
        )
    # make sure T_horizon is an int
    T_horizon = int(T_horizon)

    # Step 1: extract data from them
    t_interval, lat_interval, lon_interval = utils.simulation_utils.convert_to_lat_lon_time_bounds(
        problem.x_0,
        problem.x_T,
        deg_around_x0_xT_box=deg_around_x0_xT_box,
        temp_horizon_in_h=T_horizon,
        hours_to_hj_solve_timescale=hours_to_abs_time,
    )

    # Go over all relevant forecasts
    dict_of_relevant_fmrcs = list(
        filter(
            lambda dic: dic["t_range"][0] < problem.t_0 + timedelta(hours=T_horizon),
            problem.forecast_data_source["content"],
        )
    )

    # initialize lists for logging arrays for the individual forecasts
    RMSE_across_fmrc = []
    angle_diff_across_fmrc = []
    vec_corr_across_fmrc = []

    for fmrc_dict in dict_of_relevant_fmrcs:
        HYCOM_Forecast = xr.open_dataset(fmrc_dict["file"])
        HYCOM_Forecast = HYCOM_Forecast.fillna(0).isel(depth=0)
        # subset in time and space for mission
        HYCOM_Forecast = HYCOM_Forecast.sel(
            time=slice(
                HYCOM_Forecast["time"].data[0],
                HYCOM_Forecast["time"].data[0] + np.timedelta64(T_horizon, "h"),
            ),
            lat=slice(lat_interval[0], lat_interval[1]),
            lon=slice(lon_interval[0], lon_interval[1]),
        )
        # check if right length
        if HYCOM_Forecast["time"].shape[0] != T_horizon + 1:
            continue
        # subset Hindcast to match with Forecast
        t_interval = [HYCOM_Forecast.variables["time"][0], HYCOM_Forecast.variables["time"][-1]]
        if problem.hindcast_data_source["data_source_type"] == "cop_opendap":
            Hindcast = get_hindcast_from_copernicus(
                HYCOM_Forecast, t_interval, lat_interval, lon_interval, problem
            )
        elif problem.hindcast_data_source["data_source_type"] == "multiple_daily_nc_files":
            Hindcast = get_hindcast_from_hycom(t_interval, lat_interval, lon_interval, problem)
        else:
            raise ValueError(
                "Data source only opendap and multiple_daily_nc_files implemented right now"
            )

        # Extract data arrays for the calculation
        u_data_forecast = HYCOM_Forecast["water_u"].data
        v_data_forecast = HYCOM_Forecast["water_v"].data
        u_data_hindcast = Hindcast["water_u"].data
        v_data_hindcast = Hindcast["water_v"].data

        # Step 2: Calculate error metrics over time
        RMSE_across_fmrc.append(
            calc_speed_RMSE(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast)
        )
        angle_diff_across_fmrc.append(
            calc_abs_angle_difference(
                u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast
            )
        )
        vec_corr_across_fmrc.append(
            calc_vector_corr_over_time(
                HYCOM_Forecast.to_array().to_numpy().transpose((1, 2, 3, 0)),
                Hindcast.to_array().to_numpy().transpose((1, 2, 3, 0)),
            )
        )

    return {
        "RMSE_velocity": np.array(RMSE_across_fmrc).mean(axis=0),
        "angle_diff": np.array(angle_diff_across_fmrc).mean(axis=0),
        "vector_correlation": np.array(vec_corr_across_fmrc).mean(axis=0),
    }


def get_hindcast_from_hycom(t_interval, lat_interval, lon_interval, problem):
    # Interpolate Hindcast to the Forecast axis
    HYCOM_Hindcast = xr.open_mfdataset(
        [h_dict["file"] for h_dict in problem.hindcast_data_source["content"]]
    )
    HYCOM_Hindcast = HYCOM_Hindcast.fillna(0).isel(depth=0)
    HYCOM_Hindcast["time"] = HYCOM_Hindcast["time"].dt.round("H")
    HYCOM_Hindcast = HYCOM_Hindcast.sel(
        time=slice(t_interval[0], t_interval[1]),
        lat=slice(lat_interval[0], lat_interval[1]),
        lon=slice(lon_interval[0], lon_interval[1]),
    )
    return HYCOM_Hindcast


def get_hindcast_from_copernicus(HYCOM_Forecast, t_interval, lat_interval, lon_interval, problem):
    """Helper function to get a copernicus Hindcast frame aligned with HYCOM_Forecast."""
    t_interval[0] = t_interval[0] - np.timedelta64(3, "h")
    t_interval[1] = t_interval[1] + np.timedelta64(3, "h")
    subsetted_frame = problem.hindcast_data_source["content"].sel(
        time=slice(t_interval[0], t_interval[1]),
        latitude=slice(lat_interval[0], lat_interval[1]),
        longitude=slice(lon_interval[0], lon_interval[1]),
    )

    DS_renamed_subsetted_frame = subsetted_frame.rename(
        {"vo": "water_v", "uo": "water_u", "latitude": "lat", "longitude": "lon"}
    )
    DS_renamed_subsetted_frame = DS_renamed_subsetted_frame.fillna(0)
    Copernicus_right_time = DS_renamed_subsetted_frame.interp(time=HYCOM_Forecast["time"])
    # interpolate 2D in space
    Copernicus_H_final = Copernicus_right_time.interp(
        lon=HYCOM_Forecast["lon"].data, lat=HYCOM_Forecast["lat"].data, method="linear"
    )

    # again fill na with 0
    Copernicus_H_final = Copernicus_H_final.fillna(0)
    # return aligned dataframe
    return Copernicus_H_final


# # Note: for now just the logic for analytical currents (a bit easier)
# def calc_fmrc_errors(problem, T_horizon, deg_around_x0_xT_box, hours_to_abs_time=1):
#     # Step 0: make sure we're in analytical current case
#     if not problem.hindcast_data_source['data_source_type'] == 'analytical_function':
#         raise ValueError("problem.hindcast_data_source['data_source_type'] is not 'analytical_function'")
#     if not problem.forecast_data_source['data_source_type'] == 'analytical_function':
#         raise ValueError("problem.forecast_data_source['data_source_type'] is not 'analytical_function'")
#
#     # Step 1: extract data from them
#     t_interval, lat_interval, lon_interval = utils.simulation_utils.convert_to_lat_lon_time_bounds(
#         problem.x_0, problem.x_T,
#         deg_around_x0_xT_box=deg_around_x0_xT_box,
#         temp_horizon_in_h=T_horizon,
#         hours_to_hj_solve_timescale=hours_to_abs_time)
#
#     ana_cur = problem.hindcast_data_source['content']
#
#     # limit it to inside
#     y_interval = [max(lat_interval[0], ana_cur.spatial_domain.lo[1] + ana_cur.boundary_buffers[1]),
#                   min(lat_interval[1], ana_cur.spatial_domain.hi[1] - ana_cur.boundary_buffers[1])]
#     x_interval = [max(lon_interval[0], ana_cur.spatial_domain.lo[0] + ana_cur.boundary_buffers[0]),
#                   min(lon_interval[1], ana_cur.spatial_domain.hi[0] - ana_cur.boundary_buffers[0])]
#
#     grids_dict_hindcast, u_data_hindcast, v_data_hindcast = utils.simulation_utils.get_current_data_subset(
#         t_interval, y_interval, x_interval, problem.hindcast_data_source)
#     grids_dict_forecast, u_data_forecast, v_data_forecast = utils.simulation_utils.get_current_data_subset(
#         t_interval, y_interval, x_interval, problem.forecast_data_source)
#
#     # Step 2: Calculate things and return them as dict
#     RMSE = calc_speed_RMSE(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast)
#     angle_diff = calc_abs_angle_difference(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast)
#     vec_corr = calc_vector_correlation(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast)
#
#     return {'RMSE_velocity': RMSE, 'angle_diff': angle_diff, 'vector_correlation':vec_corr}

### HELPER FUNCTIONS
def calc_speed_RMSE(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast):
    """Helper function to calculate the RMSE on current speed."""
    RMSE_speed = np.sqrt(
        (u_data_forecast - u_data_hindcast) ** 2 + (v_data_forecast - v_data_hindcast) ** 2
    ).mean(axis=(1, 2))
    return RMSE_speed


# turns out this doesn't consider the angle so it's equivalent to RMSE...
def error_vector_magnitude(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast):
    error_vector = np.stack((u_data_hindcast, v_data_hindcast)) - np.stack(
        (u_data_forecast, v_data_forecast)
    )
    return np.linalg.norm(error_vector, axis=0).mean(axis=(1, 2))


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
