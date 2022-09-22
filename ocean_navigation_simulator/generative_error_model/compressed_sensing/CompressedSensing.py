from ocean_navigation_simulator.generative_error_model.Dataset import Dataset

import pandas as pd
import os
import numpy as np
import xarray as xr
import cvxpy as cp
import matplotlib.pyplot as plt
import modred as mr
import datetime
from typing import Tuple, List


class CompressedSensing:
    def __init__(self, forecast_dir: str, lon_range: Tuple[int, int], lat_range: Tuple[int, int]):
        self.forecast_dir = forecast_dir
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.forecast = None
        self.modes = None
        self.orig_data_shape = None

    def get_basis(self, num_modes: int, variables: Tuple[str, str] = ("utotal", "vtotal")):
        # load forecast
        files = sorted(os.listdir(self.forecast_dir))
        fc = xr.open_dataset(os.path.join(self.forecast_dir, files[0])).sel(longitude=slice(*self.lon_range),
                                                                       latitude=slice(*self.lat_range))
        if len(files) > 1:
            fc = fc.isel(time=slice(0, 24))
            for file in range(1, len(files)):
                data_temp = xr.open_dataset(os.path.join(self.forecast_dir, files[file])).sel(longitude=slice(-140, -120),
                                                                                         latitude=slice(20, 30))
                fc = xr.concat([fc, data_temp.isel(time=slice(0, 24))], dim="time")

        print(f"Loaded {round(len(fc['time']) / 24)} days of forecasts")
        self.forecast = fc

        # get most dominant modes from FC by using POD
        modes_u, eigvals_u, proj_coeffs_u, _, orig_data_shape_u = perform_POD(fc, variables[0], num_modes)
        modes_v, eigvals_v, proj_coeffs_v, _, orig_data_shape_v = perform_POD(fc, variables[1], num_modes)

        self.modes = np.array([modes_u, modes_v])
        self.orig_data_shape = orig_data_shape_u

    def perform_CS(self, buoy_data: pd.DataFrame, save_dir: str, variables: Tuple[str, str] = ("u", "v")):

        # add hour column to buoy_data
        buoy_data["hour"] = buoy_data["time"].apply(lambda x: x[:13])
        hours = sorted(set(buoy_data["hour"].tolist()))

        reconstructed_time_step = []
        for time_step in range(len(hours)):
            Psi_u, basis_coeffs_u = perform_CS(self.forecast, self.modes[0], buoy_data, variables[0], time_step=time_step)
            reconstructed_u = Psi_u @ basis_coeffs_u
            reconstructed_u = reconstructed_u.reshape(self.orig_data_shape[1:])

            Psi_v, basis_coeffs_v = perform_CS(self.forecast, self.modes[1], buoy_data, variables[1], time_step=time_step)
            reconstructed_v = Psi_v @ basis_coeffs_v
            reconstructed_v = reconstructed_v.reshape(self.orig_data_shape[1:])

            reconstructed_time_step.append([reconstructed_u, reconstructed_v])
        reconstructed_time_step = np.array(reconstructed_time_step)

        times = sorted(list(set(buoy_data["time"])))
        self._save_as_xr(reconstructed_time_step, times, save_dir)

    def _save_as_xr(self, time_steps: np.ndarray, times: List[str], save_dir: str):
        """Save the compressed sensing output in an xarray.
        """
        # store data in xarray Dataset object
        base_time = datetime.datetime.strptime(times[0], "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes=-30)
        ds = xr.Dataset(
            data_vars=dict(
                utotal=(["time", "lat", "lon"], time_steps[:, 0, :, :]),
                vtotal=(["time", "lat", "lon"], time_steps[:, 1, :, :])
            ),
            coords=dict(
                lon=np.array(np.arange(self.lon_range[0], self.lon_range[1] + 1 / 12, 1 / 12)),
                lat=np.array(np.arange(self.lat_range[0], self.lat_range[1] + 1 / 12, 1 / 12)),
                time=np.array([base_time + datetime.timedelta(hours=hours) for hours in range(len(times))])
            )
        )

        # save to file
        ds.to_netcdf(os.path.join(save_dir, f"reconstructed_{base_time}.nc"))
        print(f"Saved file for {base_time}.")


def perform_CS(forecast: xr.Dataset, modes: np.ndarray, buoy_error_data: pd.DataFrame, var: str,
               time_step: int, vis_data: bool = False, verbose: bool = False):
    # create hour column
    hour = time_step
    buoy_error_data["hour"] = buoy_error_data["time"].apply(lambda x: x[:13])
    hours = sorted(set(buoy_error_data["hour"].tolist()))
    if hour > forecast["time"].shape[0]:
        raise ValueError("Time step out of range!")
    sparse_data_time_step = buoy_error_data[buoy_error_data["hour"].values == hours[hour]]
    if verbose:
        print(f"Number of points for time step: {sparse_data_time_step.shape[0]}.")

    # get nearest grid point
    points = np.array([
        sparse_data_time_step["lon"],
        sparse_data_time_step["lat"]
    ])
    nearest_grid_points = round_to_multiple(points)

    # get idx in lon and lat
    lon_idx = np.searchsorted(forecast["longitude"].values, nearest_grid_points[0])
    lat_idx = np.searchsorted(forecast["latitude"].values, nearest_grid_points[1])

    if vis_data:
        # plot forecast and buoy positions
        plt.imshow(forecast["utotal"].isel(time=hour).squeeze(), origin="lower")
        plt.scatter(lon_idx, lat_idx, color="r")

    # get idx of points for flattened area
    flattened_idx = np.array([lat * forecast["longitude"].values.shape[0] + lon for lon, lat in zip(lon_idx, lat_idx)])

    # create the C matrix. Zero where there are no measurements and 1 where there exists a measurement
    C = np.zeros((flattened_idx.shape[0], modes.shape[0]))
    one_indices = np.array([[row, col] for row, col in zip(range(flattened_idx.shape[0]), flattened_idx)])
    C[one_indices[:, 0], one_indices[:, 1]] = 1
    # Psi matrix
    Psi = modes
    # construct A matrix
    A = C @ Psi
    # define y vector
    y = sparse_data_time_step[var].values

    # perform CS
    vx = cp.Variable(A.shape[-1])

    # # Problem formulation 1
    # objective = cp.Minimize(cp.norm(vx, 1))
    # constraints = [A @ vx == y]
    # prob = cp.Problem(objective, constraints)

    # Least squares -> might be problematic if ill-conditioned
    objective = cp.sum_squares(A @ vx - y)
    prob = cp.Problem(cp.Minimize(objective))

    # # LASSO
    # gamma = cp.Parameter(nonneg=True)
    # gamma.value = 0.05
    # error = cp.sum_squares(A @ vx - y)
    # objective = cp.Minimize(error + gamma*cp.norm(vx, 1))
    # prob = cp.Problem(objective, constraints)

    # # Tikhonov Regularization
    # gamma = cp.Parameter(nonneg=True)
    # gamma.value = 0.05
    # error = cp.sum_squares(A @ vx - y)
    # objective = cp.Minimize(error + gamma*cp.norm(vx, 2))
    # prob = cp.Problem(objective, constraints)

    prob.solve()

    if prob.status in ["infeasible", "unbounded"]:
        print(f"Problem is {prob.status}.")
        raise RuntimeError("Optimization failed and coefficients are None!")
    basis_coeffs = np.array(vx.value)
    return Psi, basis_coeffs


def perform_POD(data: xr.Dataset, var_name: str, num_modes: int):
    print(f"Percentage of NaN values: {int(100*np.isnan(data[var_name]).sum()/np.prod(data[var_name].shape))}%")
    data = data[var_name].squeeze()
    print(f"Data dims: {data.dims}, data shape: {data.values.shape}")
    data = data.values
    orig_data_shape = data.shape

    # reshape data (flattened snapshots vs time -> 2d array)
    data = data.reshape(data.shape[0], -1).T
    print(f"Shape of input data: {data.shape} [flattened space x time snapshots]")

    # compute POD: https://modred.readthedocs.io/en/stable/pod.html#modred.pod.compute_POD_arrays_snaps_method
    POD_res = mr.compute_POD_arrays_snaps_method(data, list(mr.range(num_modes)))
    # POD_res = mr.compute_POD_arrays_direct_method(data, list(mr.range(num_modes)))
    assert POD_res.eigvals.all() == np.array(sorted(POD_res.eigvals)).all(), "Eigenvalues not sorted"
    return POD_res.modes, POD_res.eigvals, POD_res.proj_coeffs, data, orig_data_shape


def round_to_multiple(numbers: np.ndarray, multiple: float = 1/12):
    return multiple * np.round_(numbers / multiple)


def vis_reconstructed_and_forecast(cs_data: xr.Dataset, fc_data: xr.Dataset):
    fig, axs = plt.subplots(5, 2, figsize=(12, 12))
    for row in range(axs.shape[0]):
        cs_frame = cs_data.isel(time=row)["utotal"]
        time = cs_data.isel(time=row)["time"].values
        time = datetime.datetime.utcfromtimestamp(int(time.astype(datetime.datetime) / 1e9))
        time = time.strftime('%Y-%m-%dT%H:%M:%S')
        img1 = axs[row, 0].imshow(cs_frame, origin="lower")
        plt.colorbar(img1, ax=axs[row, 0])
        axs[row, 0].set_title(f"GT from Compressed Sensing at t={time}")

        fc = fc_data.isel(time=row)["utotal"].squeeze()
        time = fc_data.isel(time=row)["time"].values
        time = datetime.datetime.utcfromtimestamp(int(time.astype(datetime.datetime) / 1e9))
        time = time.strftime('%Y-%m-%dT%H:%M:%S')
        img2 = axs[row, 1].imshow(fc, origin="lower")
        plt.colorbar(img2, ax=axs[row, 1])
        axs[row, 1].set_title(f"FC at t={time}")
        plt.tight_layout()
