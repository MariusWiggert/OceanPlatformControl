import numpy as np
import ocean_navigation_simulator.utils as utils

# Note: for now just the logic for analytical currents (a bit easier)
def calc_fmrc_errors(problem, T_horizon, deg_around_x0_xT_box, hours_to_abs_time=1):
    # Step 0: make sure we're in analytical current case
    if not problem.hindcast_data_source['data_source_type'] == 'analytical_function':
        raise ValueError("problem.hindcast_data_source['data_source_type'] is not 'analytical_function'")
    if not problem.forecast_data_source['data_source_type'] == 'analytical_function':
        raise ValueError("problem.forecast_data_source['data_source_type'] is not 'analytical_function'")

    # Step 1: extract data from them
    t_interval, lat_interval, lon_interval = utils.simulation_utils.convert_to_lat_lon_time_bounds(
        problem.x_0, problem.x_T,
        deg_around_x0_xT_box=deg_around_x0_xT_box,
        temp_horizon_in_h=T_horizon,
        hours_to_hj_solve_timescale=hours_to_abs_time)

    grids_dict_hindcast, u_data_hindcast, v_data_hindcast = utils.simulation_utils.get_current_data_subset(
        t_interval, lat_interval, lon_interval, problem.hindcast_data_source)
    grids_dict_forecast, u_data_forecast, v_data_forecast = utils.simulation_utils.get_current_data_subset(
        t_interval, lat_interval, lon_interval, problem.forecast_data_source)

    # Step 2: Calculate things and return them as dict
    RMSE = calc_speed_RMSE(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast)
    angle_diff = calc_abs_angle_difference(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast)
    vec_corr = calc_vector_correlation(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast)

    return {'RMSE_velocity': RMSE, 'angle_diff': angle_diff, 'vector_correlation':vec_corr}

### HELPER FUNCTIONS
def calc_speed_RMSE(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast):
    """Helper function to calculate the RMSE on current speed."""
    RMSE_speed = np.sqrt((u_data_forecast-u_data_hindcast)**2 + (v_data_forecast-v_data_hindcast)**2).mean()
    return RMSE_speed

# turns out this doesn't consider the angle so it's equivalent to RMSE...
def error_vector_magnitude(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast):
    error_vector = np.stack((u_data_hindcast, v_data_hindcast)) - np.stack((u_data_forecast, v_data_forecast))
    return np.linalg.norm(error_vector, axis=0).mean()


def calc_abs_angle_difference(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast):
    return np.abs(np.arctan2(v_data_hindcast, u_data_hindcast) - np.arctan2(v_data_forecast, u_data_forecast)).mean()


def calc_vector_correlation(u_data_forecast, v_data_forecast, u_data_hindcast, v_data_hindcast, print_out=False):
    # Flatten out the vectors
    # first Stack them
    forecast = np.stack((u_data_forecast, v_data_forecast))
    hindcast = np.stack((u_data_hindcast, v_data_hindcast))
    forecast_vec = forecast.reshape(2, -1)
    hindcast_vec = hindcast.reshape(2, -1)
    # Step 1: calculate the correlation matrix
    full_variable_vec = np.vstack((forecast_vec, hindcast_vec))
    Covariance_matrix = np.cov(full_variable_vec)
    # calculation for vector correlation
    Sigma_11 = Covariance_matrix[:2,:2]
    Sigma_22 = Covariance_matrix[2:,2:]
    Sigma_12 = Covariance_matrix[:2,2:]
    Sigma_21 = Covariance_matrix[2:,:2]
    # Matrix multiplications
    vector_correlation = np.trace(np.linalg.inv(Sigma_11) @ Sigma_12 @ np.linalg.inv(Sigma_22) @ Sigma_21)
    if print_out:
        print("vector_correlation is : ", vector_correlation)
    return vector_correlation
