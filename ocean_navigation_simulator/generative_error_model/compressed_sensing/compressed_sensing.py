from ocean_navigation_simulator.generative_error_model.Dataset import Dataset

import pandas as pd
import os
import numpy as np
import xarray as xr
import cvxpy as cp
import matplotlib.pyplot as plt
import modred as mr
import datetime


def reconstruct_currents(forecast_dir: str, buoy_data_dir: str, num_modes: int, num_time_steps: int,
                         save_dir: str, file_name: str) -> None:
    """Takes forecasts and buoy data -> performs POD -> performs CS and saves the reconstructed current
    to a netcdf file.
    """

    # load forecast
    files = sorted(os.listdir(forecast_dir))
    # TODO: use extra arguments to specify lon lat ranges.
    fc = xr.open_dataset(os.path.join(forecast_dir, files[0])).sel(longitude=slice(-140, -120),
                                                                   latitude=slice(20, 30))
    if len(files) > 1:
        fc = fc.isel(time=slice(0, 24))
        for file in range(1, len(files)):
            data_temp = xr.open_dataset(os.path.join(forecast_dir, files[file])).sel(longitude=slice(-140, -120),
                                                                                     latitude=slice(20, 30))
            fc = xr.concat([fc, data_temp.isel(time=slice(0, 24))], dim="time")

    print(f"Loaded {round(len(fc['time']) / 24)} days of forecasts")

    # get most dominant modes from FC by using POD
    modes_u, eigvals_u, proj_coeffs_u, _, orig_data_shape_u = perform_POD(fc, "utotal", num_modes)
    modes_v, eigvals_v, proj_coeffs_v, _, orig_data_shape_v = perform_POD(fc, "vtotal", num_modes)

    # load buoy data
    buoy_files = sorted(os.listdir(buoy_data_dir))
    buoy_data = pd.read_csv(os.path.join(buoy_data_dir, buoy_files[0]))
    if len(buoy_files) > 1:
        buoy_data = get_first_hours_of_df(buoy_data, num_hours=24)
        for file in range(1, len(buoy_files)-1):
            buoy_data_temp = pd.read_csv(os.path.join(buoy_data_dir, buoy_files[file]))
            buoy_data_temp = get_first_hours_of_df(buoy_data_temp, num_hours=24)
            buoy_data = pd.concat([buoy_data, buoy_data_temp], ignore_index=True)
        buoy_data_temp = pd.read_csv(os.path.join(buoy_data_dir, buoy_files[file]))
        buoy_data_temp = get_first_hours_of_df(buoy_data_temp, num_hours=24)
        buoy_data = pd.concat([buoy_data, buoy_data_temp], ignore_index=True)

    # perform CS for both u and v
    basis_coeffs_list = []
    for i in range(num_time_steps):
        Psi_u, basis_coeffs_u = perform_CS(fc, modes_u, buoy_data, "u", time_step=i, vis_data=False, verbose=False)
        Psi_v, basis_coeffs_v = perform_CS(fc, modes_v, buoy_data, "v", time_step=i, vis_data=False, verbose=False)
        basis_coeffs_list.append([basis_coeffs_u, basis_coeffs_v])
    basis_coeffs_list = np.array(basis_coeffs_list)
    time_steps = []

    # reconstruct for specified number of time steps
    for time_step in range(num_time_steps):
        reconstructed_u = Psi_u @ basis_coeffs_list[time_step, 0, :]
        reconstructed_u = reconstructed_u.reshape(orig_data_shape_u[1:])

        reconstructed_v = Psi_v @ basis_coeffs_list[time_step, 1, :]
        reconstructed_v = reconstructed_v.reshape(orig_data_shape_v[1:])
        time_steps.append([reconstructed_u, reconstructed_v])
    time_steps = np.array(time_steps)

    # store data in xarray Dataset object
    base_time = fc.isel(time=0)["time"].values
    base_time = datetime.datetime.utcfromtimestamp(int(base_time.astype(datetime.datetime)/1e9))
    # TODO: fix hardcoded range for lon and lat
    ds = xr.Dataset(
        data_vars=dict(
            utotal=(["time", "lat", "lon"], time_steps[:, 0, :, :]),
            vtotal=(["time", "lat", "lon"], time_steps[:, 1, :, :])
        ),
        coords=dict(
            lon=np.array(np.arange(-140, -120 + 1 / 12, 1 / 12)),
            lat=np.array(np.arange(20, 30 + 1 / 12, 1 / 12)),
            time=np.array([base_time + datetime.timedelta(hours=hours) for hours in range(num_time_steps)])
        )
    )

    # save to file
    ds.to_netcdf(os.path.join(save_dir, file_name))
    print("Finished computation and saved file.")


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


def get_first_hours_of_df(df: pd.DataFrame, num_hours: int) -> pd.DataFrame:
    df["hour"] = df["time"].apply(lambda x: x[:13])
    hours = sorted(set(df["hour"].tolist()))
    df = df[df["hour"].isin(hours[:24])]
    return df
