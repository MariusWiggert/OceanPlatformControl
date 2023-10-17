import os

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import dct, idct

from ocean_navigation_simulator.generative_error_model.compressed_sensing.CompressedSensing import (
    round_to_multiple,
)

# load sparse measurements
buoy_dir = "/home/jonas/Documents/Thesis/OceanPlatformControl/data/drifter_data/dataset_forecast_error/area1"
# buoy_dir = "/home/remote_jonas/OceanPlatformControl/data/drifter_data/dataset_forecast_error/area1"

file_name = sorted(os.listdir(buoy_dir))[0]
buoy_data = pd.read_csv(os.path.join(buoy_dir, file_name))

# add hour column to buoy_data
buoy_data["hour"] = buoy_data["time"].apply(lambda x: x[:13])
hours = sorted(set(buoy_data["hour"].tolist()))
buoy_data_at_time_step = buoy_data[buoy_data["hour"] == hours[0]]

# get nearest grid point
points = np.array([buoy_data_at_time_step["lon"], buoy_data_at_time_step["lat"]])
nearest_grid_points = round_to_multiple(points)

# get idx in lon and lat
lon_idx = np.searchsorted(np.arange(-140, -120, 1 / 12), nearest_grid_points[0])
lat_idx = np.searchsorted(np.arange(20, 30, 1 / 12), nearest_grid_points[1])

# get idx of points for flattened area
flattened_idx = np.array(
    [
        lat * np.array(np.arange(-140, -120, 1 / 12)).shape[0] + lon
        for lon, lat in zip(lon_idx, lat_idx)
    ]
)

y = buoy_data_at_time_step["u_error"]  # measurements
Psi = dct(np.eye(241 * 121), axis=0, norm="ortho")
Theta = Psi[flattened_idx, :]

# # cosamp
# s = cosamp(Theta, y, 10)
# reconstr = idct(s, axis=0, norm="ortho")
# plt.imshow(reconstr, origin="lower")
# plt.show()

# cvx
vx = cp.Variable(Theta.shape[-1])
objective = cp.Minimize(cp.norm(vx, 1))
constraints = [Theta @ vx == y]
prob = cp.Problem(objective, constraints)
prob.solve()

if prob.status in ["infeasible", "unbounded"]:
    print(f"Problem is {prob.status}.")
    raise RuntimeError("Optimization failed and coefficients are None!")
basis_coeffs = np.array(vx.value)

reconstr = idct(np.eye(241 * 121), axis=0, norm="ortho") @ basis_coeffs
reconstr = reconstr.reshape((121, 241))

plt.imshow(reconstr, origin="lower")
plt.show()
