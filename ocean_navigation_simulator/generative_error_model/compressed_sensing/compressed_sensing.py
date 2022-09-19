import pandas as pd
import numpy as np
import xarray as xr
import cvxpy as cp
import matplotlib.pyplot as plt
import modred as mr


def perform_CS(forecast: xr.Dataset, modes: np.ndarray, buoy_error_data: pd.DataFrame, time_step: int,
               vis_data: bool = False, verbose: bool = False):
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
    y = sparse_data_time_step["u_error"].values

    # perform CS
    vx = cp.Variable(A.shape[-1])

    # # Problem formulation 1
    # objective = cp.Minimize(cp.norm(vx, 1))
    # constraints = [A @ vx == y]
    # prob = cp.Problem(objective, constraints)

    # Just least squares
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
