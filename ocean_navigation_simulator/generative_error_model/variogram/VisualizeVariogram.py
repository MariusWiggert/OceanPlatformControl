from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram
from ocean_navigation_simulator.generative_error_model.utils import get_path_to_project

from typing import Tuple, AnyStr, Dict
import matplotlib.pyplot as plt
import numpy as np
import os


class VisualizeVariogram:
    """Loads or receives a Variogram object and visualizes the histograms and variograms
    in each dimension."""

    def __init__(self, config: Dict, variogram: Variogram=None):
        self.config = config
        self.variogram = variogram
        if variogram is not None:
            self.units = variogram.units
            self.bins_orig = variogram.bins
            self.bins_count_orig = variogram.bins_count
            self.res_orig = variogram.res_tuple
            self.variogram.is_3d = True if len(variogram.res_tuple) == 3 else False

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
            # set bin resolutions
            self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res = variogram_dict["res"]
            # compute number of bins from bins.shape
            self.variogram.lon_bins, self.variogram.lat_bins, self.variogram.t_bins = self.variogram.bins.shape[:3]
            print(f"Resolution: {[self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res]} {self.units}.")
        else:
            self.variogram.is_3d = False
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

    def plot_histograms(self, tol: Tuple[int] = (1, 1, 1), view_range: Tuple[int]=None) -> None:
        """Convenience function to handle which plotting function to call."""
        if self.variogram.is_3d:
            self.plot_histograms_3d(tol, view_range)
        else:
            self.plot_histograms_2d(tol, view_range)

    def plot_variograms(self, variable: AnyStr="u", tol: Tuple[int] = (1, 1, 1), view_range: Tuple[int]=None) -> None:
        """Convenience function to handle which plotting function to call."""
        if self.variogram.is_3d:
            self.plot_variograms_3d(variable, tol, view_range)
        else:
            self.plot_variograms_2d(variable, tol, view_range)

    def plot_histograms_3d(self, tol: Tuple[int]=None, view_range: Tuple[int]=None) -> None:
        """Plots the histogram of bins in each axis [lon, lat, time].
        
        tol -        the slicing tolerance when plotting over one dimension. If the variogram was
                     infinitely dense, ideally this would be one.
        view_range - three dim list which defines the index per axis used as a cut-off when
                     visualizing the sliced plots.
        """

        tol, view_range = self._plot_input_checker(tol, view_range)
        view_indices = self._get_view_indices(view_range)

        # plot histogram for each axis
        fig, axs = plt.subplots(1, 3, figsize=(25, 10))

        lon_x = self.variogram.lon_res*(np.arange(self.variogram.lon_bins)[:view_indices[0]])
        lon_y = np.sum(self.variogram.bins_count[:, :tol[1], :tol[2], 0], axis=(1, 2))[:view_indices[0]]
        axs[0].bar(lon_x, lon_y, width=self.variogram.lon_res-0.1, align="edge")
        axs[0].set_xlabel(f"Lon [{self.units}]")
        axs[0].set_ylabel("Frequency")
        axs[0].set_xlim(left=0)

        lat_x = self.variogram.lat_res*(np.arange(self.variogram.lat_bins)[:view_indices[1]])
        lat_y = np.sum(self.variogram.bins_count[:tol[0], :, :tol[2], 0], axis=(0, 2))[:view_indices[1]]
        axs[1].bar(lat_x, lat_y, width=self.variogram.lat_res-0.1, align="edge")
        axs[1].set_xlabel(f"Lat [{self.units}]")
        axs[1].set_xlim(left=0)

        t_x = self.variogram.t_res*(np.arange(self.variogram.t_bins)[:view_indices[2]])
        t_y = np.sum(self.variogram.bins_count[:tol[0], :tol[1], :, 0], axis=(0, 1))[:view_indices[2]]
        axs[2].bar(t_x, t_y, width=self.variogram.t_res-0.1, align="edge")
        axs[2].set_xlabel("Time [hrs]")
        axs[2].set_xlim(left=0)
        plt.show()

    def plot_histograms_2d(self, tol: Tuple[int] = None, view_range: Tuple[int] = None) -> None:
        """Plots the histogram of bins in each axis [lon, lat, time].

        tol -        the slicing tolerance when plotting over one dimension. If the variogram was
                     infinitely dense, ideally this would be one.
        view_range - three dim list which defines the index per axis used as a cut-off when
                     visualizing the sliced plots.
        """

        tol, view_range = self._plot_input_checker(tol, view_range)
        view_indices = self._get_view_indices(view_range)

        # plot histogram for each axis
        fig, axs = plt.subplots(1, 2, figsize=(25, 10))

        space_x = self.variogram.space_res * (np.arange(self.variogram.space_bins)[:view_indices[0]])
        space_y = np.sum(self.variogram.bins_count[:, :tol[1], 0], axis=1)[:view_indices[0]]
        axs[0].bar(space_x, space_y, width=self.variogram.space_res - 0.1, align="edge")
        axs[0].set_xlabel(f"Distance (Euclidean) [{self.units}]")
        axs[0].set_ylabel("Frequency")
        axs[0].set_xlim(left=0)

        t_x = self.variogram.t_res * (np.arange(self.variogram.t_bins)[:view_indices[1]])
        t_y = np.sum(self.variogram.bins_count[:tol[0], :, 0], axis=0)[:view_indices[1]]
        axs[1].bar(t_x, t_y, width=self.variogram.t_res - 0.1, align="edge")
        axs[1].set_xlabel("Time [hrs]")
        axs[1].set_xlim(left=0)
        plt.show()

    def plot_variograms_3d(self, variable: AnyStr="u", tol: Tuple[int]=None, view_range: Tuple[int]=None) -> None:
        """Plots the sliced variogram for each axis [lon, lat, time].
        
        tol -        the slicing tolerance when plotting over one dimension. If the variogram was
                     infinitely dense, ideally this would be one.
        view_range - three dim list which defines the index per axis used as a cut-off when
                     visualizing the sliced plots.
        """

        tol, view_range = self._plot_input_checker(tol, view_range)
        view_indices = self._get_view_indices(view_range)
        
        variable_map = {"u": 0, "v": 1}
        try:
            var = variable_map[variable]
        except:
            raise ValueError("Specified variable does not exist")

        # (Note: division needed to normalize by dims of other bins)
        fig, axs = plt.subplots(1, 3, figsize=(25, 10))

        # Only divide if denom is non-zero, else zero (https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero)
        lon_y_num = np.sum(self.variogram.bins[:, :tol[1], :tol[2], var], axis=(1, 2))[:view_indices[0]]
        # lon_y_denom = self.variogram.bins.shape[1]*self.variogram.bins.shape[2]
        lon_y_denom = tol[1]*tol[2]
        axs[0].scatter(self.variogram.lon_res*(np.arange(self.variogram.lon_bins)[:view_indices[0]]) + self.variogram.lon_res,\
            np.divide(lon_y_num, lon_y_denom, out=np.zeros_like(lon_y_num), where=lon_y_denom != 0), marker="x")
        axs[0].set_xlabel(f"Lon lag [{self.units}]")
        axs[0].set_ylabel("Semivariance")
        axs[0].set_xlim(left=0)
        axs[0].set_ylim([0, 1.5])

        lat_y_num = np.sum(self.variogram.bins[:tol[0], :, :tol[2], var], axis=(0, 2))[:view_indices[1]]
        # lat_y_denom = self.variogram.bins.shape[0]*self.variogram.bins.shape[2]
        lat_y_denom = tol[0]*tol[2]
        axs[1].scatter(self.variogram.lat_res*(np.arange(self.variogram.lat_bins)[:view_indices[1]]) + self.variogram.lat_res,\
            np.divide(lat_y_num, lat_y_denom, out=np.zeros_like(lat_y_num), where=lat_y_denom != 0), marker="x")
        axs[1].set_xlabel(f"Lat lag [{self.units}]")
        axs[1].set_xlim(left=0)
        axs[1].set_ylim([0, 1.5])

        t_y_num = np.sum(self.variogram.bins[:tol[0], :tol[1], :, var], axis=(0, 1))[:view_indices[2]]
        # t_y_denom = self.variogram.bins.shape[0]*self.variogram.bins.shape[1]
        t_y_denom = tol[0]*tol[1]
        axs[2].scatter(self.variogram.t_res*(np.arange(self.variogram.t_bins)[:view_indices[2]]) + self.variogram.t_res,\
            np.divide(t_y_num, t_y_denom, out=np.zeros_like(t_y_num), where=t_y_denom != 0), marker="x")
        axs[2].set_xlabel("Time lag [hrs]")
        axs[2].set_xlim(left=0)
        axs[2].set_ylim([0, 1.5])
        plt.show()

    def plot_variograms_2d(self, variable: AnyStr = "u", tol: Tuple[int] = None, view_range: Tuple[int] = None) -> None:
        """Plots the sliced variogram for each axis [lon, lat, time].

        tol -        the slicing tolerance when plotting over one dimension. If the variogram was
                     infinitely dense, ideally this would be one.
        view_range - three dim list which defines the index per axis used as a cut-off when
                     visualizing the sliced plots.
        """

        tol, view_range = self._plot_input_checker(tol, view_range)
        view_indices = self._get_view_indices(view_range)

        variable_map = {"u": 0, "v": 1}
        try:
            var = variable_map[variable]
        except:
            raise ValueError("Specified variable does not exist")

        # (Note: division needed to normalize by dims of other bins)
        fig, axs = plt.subplots(1, 2, figsize=(25, 10))

        # Only divide if denom is non-zero, else zero (https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero)
        space_y_num = np.sum(self.variogram.bins[:, :tol[1], var], axis=1)[:view_indices[0]]
        space_y_denom = tol[1]
        axs[0].scatter(self.variogram.space_res * (np.arange(self.variogram.space_bins)[:view_indices[0]]) + self.variogram.space_res,
                       np.divide(space_y_num, space_y_denom, out=np.zeros_like(space_y_num), where=space_y_denom != 0), marker="x")
        axs[0].set_xlabel(f"Distance (Euclidean) lag [{self.units}]")
        axs[0].set_ylabel("Semivariance")
        axs[0].set_xlim(left=0)
        axs[0].set_ylim([0, 1.5])

        t_y_num = np.sum(self.variogram.bins[:tol[0], :, var], axis=0)[:view_indices[1]]
        t_y_denom = tol[0]
        axs[1].scatter(self.variogram.t_res * (np.arange(self.variogram.t_bins)[:view_indices[1]]) + self.variogram.t_res,
                       np.divide(t_y_num, t_y_denom, out=np.zeros_like(t_y_num), where=t_y_denom != 0), marker="x")
        axs[1].set_xlabel("Time lag [hrs]")
        axs[1].set_xlim(left=0)
        axs[1].set_ylim([0, 1.5])
        plt.show()

    def _plot_input_checker(self, tol: Tuple[int] = None, view_range: Tuple[int] = None):
        if self.variogram.bins is None:
            raise Exception("Need to run build_variogram() first!")
        if tol is None:
            tol = self.variogram.bins.shape[:-1]
        if view_range is None:
            view_range = np.array(self.variogram.bins.shape[:-1]) * np.array(self.variogram.res_tuple)
        return tol, view_range

    def _get_view_indices(self, view_range: Tuple[int]):
        """Calculates the indices for each axis for a given view_range."""
        bin_sizes = self.variogram.bins.shape[:-1]
        view_indices = []
        for i in range(len(self.variogram.res_tuple)):
            idx = int(view_range[i]/self.variogram.res_tuple[i])
            if idx <= bin_sizes[i]:
                view_indices.append(idx)
            else:
                raise ValueError("View range is too large!")
        return view_indices

    def plot_hist_vario_for_axis(self, axis_name: str="time", variable: str="u", cutoff=5000) -> None:
        """Function for convenience when focusing on a single variable."""

        if self.variogram.bins is None:
            raise Exception("Need to run build_variogram() first!")

        variable_map = {"u": 0, "v": 1}
        try:
            var = variable_map[variable]
        except:
            raise ValueError("Specified variable does not exist")

        axis_map = {"lon": {"bins":self.variogram.lon_bins, "res":self.variogram.lon_res, "other_axes":(1,2)},
                    "lat": {"bins":self.variogram.lat_bins, "res":self.variogram.lat_res, "other_axes":(0,2)},
                    "time": {"bins":self.variogram.t_bins, "res":self.variogram.t_res, "other_axes":(0,1)}}

        fig, axs = plt.subplots(2,1,figsize=(20,20))
        axs[0].bar(np.arange(axis_map[axis_name]["bins"])[:cutoff]*axis_map[axis_name]["res"], np.sum(self.variogram.bins_count[:,:,:,0], axis=axis_map[axis_name]["other_axes"])[:cutoff])
        axs[0].title.set_text(f"{axis_name}")
        var_y_num = np.sum(self.variogram.bins[:, :, :, var], axis=axis_map[axis_name]["other_axes"])[:cutoff]
        # var_y_denom = np.sum(self.variogram.bins_count[:,:,:,var], axis=axis_map[axis_name]["other_axes"])[:cutoff]
        var_y_denom = self.variogram.bins.shape[axis_map[axis_name]["other_axes"][0]]*self.variogram.bins.shape[axis_map[axis_name]["other_axes"][1]]
        axs[1].scatter(np.arange(axis_map[axis_name]["bins"])[:cutoff]*axis_map[axis_name]["res"], var_y_num/var_y_denom, marker="x")
        axs[1].title.set_text(f"{axis_name}")
        plt.show()


def load_variogram_from_npy(file_path: str):
    data = np.load(file_path, allow_pickle=True)
    variogram_dict = {
        "bins": data.item().get("bins"),
        "bins_count": data.item().get("bins_count"),
        "res": data.item().get("res"),
        "units": data.item().get("units"),
        "detrend_metrics": list(data.item().get("detrend_metrics"))
    }
    return variogram_dict
   