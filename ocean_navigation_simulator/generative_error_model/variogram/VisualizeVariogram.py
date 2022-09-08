from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram
from ocean_navigation_simulator.generative_error_model.utils import get_path_to_project

from typing import Tuple, AnyStr, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class VisualizeVariogram:
    """Loads or receives a Variogram object and visualizes the histograms and variograms
    in each dimension."""

    def __init__(self, config: Dict, variogram: Variogram = None):
        self.config = config
        self.variogram = variogram
        self.vars = None
        if self.variogram is not None:
            self.units = variogram.units
            self.bins_orig = variogram.bins
            self.bins_count_orig = variogram.bins_count
            self.res_orig = variogram.res_tuple
            self.variogram.is_3d = True if len(variogram.res_tuple) == 3 else False
            if self.variogram.is_3d:
                self.vars = ["lon_lag", "lat_lag", "t_lag"]
            else:
                self.vars = ["space_lag", "t_lag"]


        self.project_path = get_path_to_project(os.getcwd())

    def read_variogram_from_file(self, file_name: str=None):
        """Reads variogram data from file and creates a Variogram object."""

        variogram_dir = os.path.join(self.project_path, self.config["data_dir"], "variogram")
        variogram_file = os.path.join(variogram_dir, file_name)
        if not os.path.exists(variogram_file):
            raise ValueError(f"Variogram file '{variogram_file.split('/')[-1]}' does not exist!")
        print(f"Loaded variogram from: {variogram_file.split('/')[-1]}.")

        # initialize empty variogram and add bins and bins_count
        self.variogram = Variogram()
        variogram_dict = load_variogram_from_npy(variogram_file)
        self.variogram.bins, self.variogram.bins_count = variogram_dict["bins"], variogram_dict["bins_count"]
        self.units = variogram_dict["units"]

        # different setup for 3d or 2d variogram
        if len(variogram_dict["res"]) == 3:
            self.variogram.is_3d = True
            self.vars = ["lon_lag", "lat_lag", "t_lag"]
            # set bin resolutions
            self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res = variogram_dict["res"]
            # compute number of bins from bins.shape
            self.variogram.lon_bins, self.variogram.lat_bins, self.variogram.t_bins = self.variogram.bins.shape[:3]
            print(f"Resolution: {[self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res]} {self.units}.")
        else:
            self.variogram.is_3d = False
            self.vars = ["space_lag", "t_lag"]
            # set bin resolutions
            self.variogram.space_res, self.variogram.t_res = variogram_dict["res"]
            # compute number of bins from bins.shape
            self.variogram.space_bins, self.variogram.t_bins = self.variogram.bins.shape[:2]
            print(f"Resolution: {[self.variogram.space_res, self.variogram.t_res]} {self.units}.")

        self.variogram.res_tuple = variogram_dict["res"]
        # add detrend statistics to variogram
        self.variogram.detrend_metrics = variogram_dict["detrend_metrics"]
        print(f"Number of pairs: {int(self.variogram.bins_count.sum()/2)}.")

        # save original arrays to allow for increasing/decreasing res
        self.bins_orig = self.variogram.bins
        self.bins_count_orig = self.variogram.bins_count
        self.res_orig = self.variogram.res_tuple

    def decrease_variogram_res(self, res_tuple: Tuple[int]):
        """Reduces the resolution of the variogram without the need to recompute."""

        # revert to original variogram data to allow decreasing bin sizes (to min size)
        self.variogram.bins = self.bins_orig
        self.variogram.bins_count = self.bins_count_orig

        # check whether res_tuple has the correct length given dimenson of variogram
        if (len(res_tuple) == 3 and self.variogram.is_3d is False) or (len(res_tuple) == 2 and self.variogram.is_3d is True):
            raise ValueError("Length of 'res_tuple' does not match dimension of Variogram!")

        # check whether not all new resolutions are a multiple of the original resolution
        if not all(np.mod(res_tuple, self.res_orig) == 0):
            raise ValueError(f"Make sure the specified resolutions are multiples of the original resolution: {self.res_orig}.")

        # instead of weighted sum -> use bins_count to first multiply bins
        self.variogram.bins = self.variogram.bins * self.variogram.bins_count

        # figure out scaling of new variogram wrt old variogram
        scaling_temp = []
        for i in range(len(res_tuple)):
            scaling_temp.append(np.round(res_tuple[i]/self.res_orig[i]))
        scaling_vec = np.array(scaling_temp, dtype=np.int32)

        # pad array such that it can be resized appropriately
        self.variogram.bins = self._make_arr_size_divisible(self.variogram.bins, scaling_vec)
        self.variogram.bins_count = self._make_arr_size_divisible(self.variogram.bins_count, scaling_vec)

        # rescale axes one by one
        for dim in range(len(res_tuple)):
            self.variogram.bins = self._decrease_res_in_axis(self.variogram.bins, axis=dim, scaling=scaling_vec[dim])
            self.variogram.bins_count = self._decrease_res_in_axis(self.variogram.bins_count, axis=dim, scaling=scaling_vec[dim])

        # write new resolutions for plotting
        if len(res_tuple) == 3:
            self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res = res_tuple
            self.variogram.lon_bins, self.variogram.lat_bins, self.variogram.t_bins = self.variogram.bins.shape[:3]
        else:
            self.variogram.space_res, self.variogram.t_res = res_tuple
            self.variogram.space_bins, self.variogram.t_bins = self.variogram.bins.shape[:2]
        self.variogram.res_tuple = res_tuple

        # normalize by frequency per bins
        self.variogram.bins = np.divide(self.variogram.bins, self.variogram.bins_count,
                                        out=np.zeros_like(self.variogram.bins), where=self.variogram.bins_count != 0)

    def _decrease_res_in_axis(self, arr: np.ndarray, axis: int, scaling: int):
        """Sums blocks of dims of size scaling over the specified axis"""
        # Idea: https://stackoverflow.com/questions/47428193/summing-blocks-of-n-rows-in-numpy-array

        # get shape of incoming array
        temp_shape = list(arr.shape)
        # add an axis behind the reduced axis of dim=scaling
        temp_shape.insert(axis+1, scaling)
        # reduce axis dimension by scaling factor
        temp_shape[axis] = int(arr.shape[axis]/scaling)
        # reshape array according to calculated shape and sum over newly introduced axis
        arr = arr.reshape(temp_shape).sum(axis=axis+1)
        return arr

    def _make_arr_size_divisible(self, arr: np.ndarray, scaling_vec: Tuple[int]):
        """Takes bins or bins_count and makes sure the first two/three dimensions
        of the array divide by the scaling tuple. This is needed when changing the
        resolution of the bins."""

        for dim, scaling in enumerate(scaling_vec):
            dim_extension = (scaling - arr.shape[dim] % scaling) % scaling

            dims_map_3d = {0: (dim_extension, arr.shape[1], arr.shape[2], 2),
                           1: (arr.shape[0], dim_extension, arr.shape[2], 2),
                           2: (arr.shape[0], arr.shape[1], dim_extension, 2)}
            dims_map_2d = {0: (dim_extension, arr.shape[1], 2),
                           1: (arr.shape[0], dim_extension, 2)}
            # maps dimension to right mapping
            dims_map = {3: dims_map_3d, 2: dims_map_2d}

            if dim_extension != 0:
                new_zero_array = np.zeros(dims_map[len(scaling_vec)][dim])
                arr = np.concatenate((arr, new_zero_array), axis=dim)
        return arr

    def save_variogram(self, file_path: str) -> None:
        """Saves entire variogram data to file.
        """
        if self.variogram.bins is None:
            raise Exception("Need to build variogram first before you can save it!")

        data_to_save = {"bins": self.variogram.bins,
                        "bins_count": self.variogram.bins_count,
                        "res": self.variogram.res_tuple,
                        "units": self.variogram.units,
                        "detrend_metrics": self.variogram.detrend_metrics
                        }
        np.save(file_path, data_to_save)
        print(f"\nSaved variogram data to: {file_path}")

    def save_tuned_variogram_2d(self, view_range: List[int], file_path: str):
        """Converts hand-tuned variogram to dataframe and saves it.
        """
        space_lag, time_lag = [], []
        u_semivariance, v_semivariance = [], []
        for space in range(int(view_range[0]/self.variogram.res_tuple[0])):
            for time in range(int(view_range[1]/self.variogram.res_tuple[1])):
                u_semivariance.append(self.variogram.bins[space, time, 0])
                v_semivariance.append(self.variogram.bins[space, time, 1])
                space_lag.append((space+1)*self.variogram.space_res)
                time_lag.append((time+1)*self.variogram.t_res)

        df = pd.DataFrame({"space_lag": space_lag,
                           "t_lag": time_lag,
                           "u_semivariance": u_semivariance,
                           "v_semivariance": v_semivariance,
                           "detrend_u": [self.variogram.detrend_metrics["u_error"] for _ in range(len(space_lag))],
                           "detrend_v": [self.variogram.detrend_metrics["v_error"] for _ in range(len(space_lag))]})
        df.to_csv(file_path, index=False)
        print(f"Saved hand-tuned variogram at {file_path}.")

    def plot_histograms(self, tol: Tuple[int] = (1, 1, 1), view_range: Tuple[int] = None) -> None:
        """Plots the histogram of bins in each axis [lon, lat, time] in 3d or [space, time] in 2d.

        tol -        the slicing tolerance when plotting over one dimension. If the variogram was
                     infinitely dense, ideally this would be one.
        view_range - three dim list which defines the index per axis used as a cut-off when
                     visualizing the sliced plots.
        """

        figure, axs = plt.subplots(1, len(self.vars), figsize=(25, 10))
        for idx, var in enumerate(self.vars):
            self._plot_sliced_histogram(var, idx, tol, view_range, axs[idx])
        plt.show()

    def _plot_sliced_histogram(self, var: str, idx: int, tol: Tuple[int], view_range: Tuple[int], ax: plt.axis) -> plt.axis:
        tol, view_range = self._plot_input_checker(idx, tol, view_range)
        view_index = self._get_view_index(idx, view_range)
        bin_sizes = self.variogram.bins.shape[:-1]

        var_x = self.variogram.res_tuple[idx] * (np.arange(bin_sizes[idx])[:view_index])
        if self.variogram.is_3d:
            other_indices = [0, 1, 2]
            del other_indices[idx]
            var_y = np.sum(self.variogram.bins_count[:tol[0], :tol[1], :tol[2], 0], axis=tuple(other_indices))[:view_index]
        else:
            other_indices = [0, 1]
            del other_indices[idx]
            var_y = np.sum(self.variogram.bins_count[:tol[0], :tol[1], 0], axis=tuple(other_indices))[:view_index]

        ax.bar(var_x, var_y, width=self.variogram.res_tuple[idx] - 0.1, align="edge")
        units = self.units
        if var == "t_lag":
            units = "hrs"
        ax.set_xlabel(f"{var} [{units}]")
        ax.set_ylabel("Frequency")
        ax.set_xlim(left=0)

    def plot_variograms(self, tol: Tuple[int] = (1, 1, 1), view_range: Tuple[int] = None, error_variable: AnyStr = "u") -> None:
        """Plots the sliced variogram for each axis [lon, lat, time] in 3d or [space, time] in 2d.

        tol -        the slicing tolerance when plotting over one dimension. If the variogram was
                     infinitely dense, ideally this would be one.
        view_range - three dim list which defines the index per axis used as a cut-off when
                     visualizing the sliced plots.
        """
        figure, axs = plt.subplots(1, len(self.vars), figsize=(25, 10))
        for idx, var in enumerate(self.vars):
            self._plot_sliced_variogram(var, idx, tol, view_range, axs[idx], error_variable)
        plt.show()

    def _plot_sliced_variogram(self, var: str, idx: int, tol: Tuple[int], view_range: Tuple[int], ax: plt.axis, error_variable: AnyStr) -> plt.axis:
        tol, view_range = self._plot_input_checker(idx, tol, view_range)
        view_index = self._get_view_index(idx, view_range)
        bin_sizes = self.variogram.bins.shape[:-1]

        error_variable_map = {"u": 0, "v": 1}
        try:
            error_var = error_variable_map[error_variable]
        except:
            raise ValueError("Specified variable does not exist")

        # Only divide if denom is non-zero, else zero (https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero)
        if self.variogram.is_3d:
            other_indices = [0, 1, 2]
            del other_indices[idx]
            var_y_num = np.sum(self.variogram.bins[:tol[0], :tol[1], :tol[2], error_var], axis=tuple(other_indices))[:view_index]
            var_y_denom = tol[other_indices[0]] * tol[other_indices[1]]
        else:
            other_indices = [0, 1]
            del other_indices[idx]
            var_y_num = np.sum(self.variogram.bins[:tol[0], :tol[1], error_var], axis=tuple(other_indices))[:view_index]
            var_y_denom = tol[other_indices[0]]

        ax.scatter(
            self.variogram.res_tuple[idx] * (np.arange(bin_sizes[idx])[:view_index]) + self.variogram.res_tuple[idx], \
            np.divide(var_y_num, var_y_denom, out=np.zeros_like(var_y_num), where=var_y_denom != 0), marker="x")
        units = self.units
        if var == "t_lag":
            units = "hrs"
        ax.set_xlabel(f"{var} [{units}]")
        ax.set_ylabel("Semivariance")
        ax.set_xlim(left=0)
        ax.set_ylim([0, 1.5])

    def _plot_input_checker(self, idx: int, tol: Tuple[int] = None, view_range: Tuple[int] = None):
        if self.variogram.bins is None:
            raise Exception("Need to run build_variogram() first!")
        if tol is None:
            tol = self.variogram.bins.shape[:-1]
        tol = list(tol)
        tol[idx] = self.variogram.bins.shape[idx]
        if view_range is None:
            view_range = np.array(self.variogram.bins.shape[:-1]) * np.array(self.variogram.res_tuple)
        return tol, view_range

    def _get_view_index(self, idx: int, view_range: Tuple[int]):
        """Calculates the indices for each axis for a given view_range."""
        num_bin = self.variogram.bins.shape[idx]
        bin_idx = int(view_range[idx]/self.variogram.res_tuple[idx])
        if bin_idx <= num_bin:
            view_index = bin_idx
        else:
            raise ValueError("View range is too large!")
        return view_index


def load_variogram_from_npy(file_path: str):
    data = np.load(file_path, allow_pickle=True)
    variogram_dict = {
        "bins": data.item().get("bins"),
        "bins_count": data.item().get("bins_count"),
        "res": data.item().get("res"),
        "units": data.item().get("units"),
        "detrend_metrics": list(data.item().get("detrend_metrics").values())[0]
    }
    return variogram_dict
   