"""
All the metrics for used for evaluating the forecast - buoy error
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_all_metrics():
    # TODO: function to compute all metrics and plot them
    fig, axs = plt.subplots(3,figsize=(20,6))
    fig.suptitle("Ocean Current Error Metrics")
    axs[0].plot()
    pass

def calc_speed_mean(u_data_hindcast, v_data_hindcast, u_data_measured, v_data_measured):
    mean_speed = (u_data_hindcast - u_data_measured)*np.cos(np.pi/4) + (v_data_hindcast - v_data_measured)*np.cos(np.pi/4)
    return mean_speed

def calc_speed_RMSE(u_data_hindcast, v_data_hindcast, u_data_measured, v_data_measured):
    RMSE_speed = np.sqrt((u_data_hindcast-u_data_measured)**2 + (v_data_hindcast-v_data_measured)**2)
    return RMSE_speed

def calc_vector_correlation(u_data_hindcast, v_data_hindcast, u_data_measured, v_data_measured, print_out=False):
    """
    Calculates the vector correlation for one buoy for a specific day,
    where one day typically has 24 measurements.
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
    Sigma_11 = Covariance_matrix[:2,:2]
    Sigma_22 = Covariance_matrix[2:,2:]
    Sigma_12 = Covariance_matrix[:2,2:]
    Sigma_21 = Covariance_matrix[2:,:2]
    # Matrix multiplications
    epsilon = 0
    try:
        vector_correlation = np.trace(np.linalg.inv(Sigma_11) @ Sigma_12 @ np.linalg.inv(Sigma_22) @ Sigma_21)
    except:
        vector_correlation = np.NaN
        while np.isnan(vector_correlation):
            epsilon += 5e-5
            vector_correlation = np.trace(np.linalg.inv(Sigma_11 + epsilon*np.eye(2)) @ Sigma_12 @ np.linalg.inv(Sigma_22 + epsilon*np.eye(2)) @ Sigma_21)
            if epsilon > 1e-1:
                break
    return vector_correlation

def get_vector_correlation_per_day(df_day):
    """
    Calculates the vector correlation per day for each buoy and takes the average over all buoys

    Expects a dataframe with data from one or more buoys over one day
    """
    buoy_names = set(df_day["buoy"].tolist())
    vec_corr_total = 0
    for name in buoy_names:
        points_buoy = df_day[df_day["buoy"] == name]
        vec_corr = calc_vector_correlation(points_buoy["u_hindcast"],
                                                points_buoy["v_hindcast"],
                                                points_buoy["u"],
                                                points_buoy["v"])
        if np.isnan(vec_corr):
            vec_corr = 0
        vec_corr_total += vec_corr
    if len(buoy_names) != 0:
        return vec_corr_total/len(buoy_names)
    return -1

def get_vector_correlation_over_time(df):
    """
    Gets the vector correlation over the entire time range covered in the dataframe
    """
    df["day"] = df["time"].apply(lambda x: x[:10])
    days = sorted(set(df["day"].tolist()))
    vec_corr = []
    for day in days:
        df_day = df[df["day"] == day]
        vec_corr.append(get_vector_correlation_per_day(df_day))
    df_vec_corr = pd.DataFrame({"day": days, "vec_corr": vec_corr})
    return df_vec_corr

def plot_metric(time, metric, supress_nth_label = 24):
    fig, ax = plt.subplots(figsize=(20,6))
    plt.plot(time, metric)

    # to supress most labels
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % supress_nth_label != 0:
            label.set_visible(False)
    plt.show()