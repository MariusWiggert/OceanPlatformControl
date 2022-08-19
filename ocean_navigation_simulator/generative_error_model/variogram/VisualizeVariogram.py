from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram
from ocean_navigation_simulator.generative_error_model.utils import load_config

from typing import Tuple, List, AnyStr
import matplotlib.pyplot as plt
import numpy as np
import os


class VisualizeVariogram:

    def __init__(self, variogram:Variogram=None):
        self.variogram = variogram
        if variogram is not None:
            self.units = variogram.units


    def read_variogram_from_file(self, file_name: str=None):
        """Reads variogram data from file and creates a Variogram object."""

        self.config = load_config()["variogram"]
        if file_name is None:
            file_path = os.path.join(self.config["data_dir"], self.config["file_name"])
        else:
            file_path = os.path.join(self.config["data_dir"], file_name)

        # initialize empty variogram and add bins and bins_count
        self.variogram = Variogram()
        variogram_data = load_variogram_from_npy(file_path)
        self.variogram.bins, self.variogram.bins_count = variogram_data[:2]
        self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res = variogram_data[2]

        # compute lon bins from bins.shape
        self.variogram.lon_bins, self.variogram.lat_bins, self.variogram.t_bins = self.variogram.bins.shape[:3]

        # checking if units were set when saving, otherwise use default
        if variogram_data[3] is None:
            self.units = "km"
        else:
            self.units = variogram_data[3]

        print(f"Loaded variogram from: {file_path.split('/')[-1]}.")
        print(f"Resolution: {[self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res]} {self.units}.")
        print(f"Number of pairs: {int(self.variogram.bins_count.sum()/2)}.")


    def decrease_variogram_res(self, res_tuple: Tuple[int]):
        """Reduces the resolution of the variogram without the need to recompute."""

        current_res = np.array([self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res])
        if np.less(np.array(res_tuple), current_res).any() == True:
            raise ValueError(f"Make sure the specified resolutions are multiples of the original resolution: {current_res}.")

        # instead of weighted sum -> use bins_count to first multiply bins
        self.variogram.bins = self.variogram.bins * self.variogram.bins_count

        scaling_vec = np.array([np.round(res_tuple[0]/self.variogram.lon_res),
                        np.round(res_tuple[1]/self.variogram.lat_res),
                        np.round(res_tuple[2]/self.variogram.t_res)], dtype=np.int32)

        # pad array such that it can be resized appropriately
        self.variogram.bins = self._make_arr_size_divisible(self.variogram.bins, scaling_vec)
        self.variogram.bins_count = self._make_arr_size_divisible(self.variogram.bins_count, scaling_vec)

        # rescale axes one by one
        for i, dim in enumerate(range(3)):
            self.variogram.bins = self._decrease_res_in_axis(self.variogram.bins, axis=dim, scaling=scaling_vec[i])
            self.variogram.bins_count = self._decrease_res_in_axis(self.variogram.bins_count, axis=dim, scaling=scaling_vec[i])
            
        self.variogram.lon_res, self.variogram.lat_res, self.variogram.t_res = res_tuple
        self.variogram.lon_bins, self.variogram.lat_bins, self.variogram.t_bins = self.variogram.bins.shape[:3]

        # normalize by frequency per bins
        self.variogram.bins = np.divide(self.variogram.bins, self.variogram.bins_count,\
            out=np.zeros_like(self.variogram.bins), where=self.variogram.bins_count!=0)


    def _decrease_res_in_axis(self, arr: np.ndarray, axis:int, scaling: int):
        """Sums blocks of dims of size scaling over the specified axis"""
        # Idea: https://stackoverflow.com/questions/47428193/summing-blocks-of-n-rows-in-numpy-array

        # get shape of incoming array
        temp_shape = list(arr.shape)
        # add an axis behind the reduced axis of dim=scaling
        temp_shape.insert(axis+1, scaling)
        # reduce axis dimension by scaling factor
        temp_shape[axis] = int(arr.shape[axis]/scaling)
        # reshape array according to calculated shape and sum over newly introduced axis
        arr = arr.reshape((temp_shape)).sum(axis=axis+1)
        return arr


    def _make_arr_size_divisible(self, arr: np.ndarray, scaling_vec: Tuple[int]):
        """Takes bins or bins_count and makes sure the first three dimensions
        of the array divide by the scaling tuple. This is needed when changing the
        resolution of the bins."""

        for dim, scaling in zip(range(3), scaling_vec):
            dim_extension = (scaling - arr.shape[dim]%scaling)%scaling
            if dim_extension != 0:
                other_dims_map = {0: (dim_extension, arr.shape[1], arr.shape[2], 2),
                    1: (arr.shape[0], dim_extension ,arr.shape[2] ,2),
                    2: (arr.shape[0], arr.shape[1], dim_extension, 2)}
                new_zero_array = np.zeros(other_dims_map[dim])
                arr = np.concatenate((arr, new_zero_array), axis=dim)
        return arr
        

    def plot_histograms(self, tol: Tuple[int]=None, view_range: List[int]=None) -> None:
        """Plots the histogram of bins in each axis [lon, lat, time].
        
        tol -        the slicing tolerance when plotting over one dimension. If the variogram was
                     infinitely dense, ideally this would be one.
        view_range - three dim list which defines the index per axis used as a cut-off when
                     visualizing the sliced plots.
        """

        if self.variogram.bins is None:
            raise Exception("Need to run build_variogram() first!")
        if tol is None:
            tol = self.variogram.bins.shape[:3]
        if view_range is None:
            view_range = self.variogram.bins.shape[:3]

        # plot histogram for each axis
        fig, axs = plt.subplots(1,3,figsize=(25,10))

        lon_x = self.variogram.lon_res*(np.arange(self.variogram.lon_bins)[:view_range[0]])
        lon_y = np.sum(self.variogram.bins_count[:,:tol[1],:tol[2],0], axis=(1,2))[:view_range[0]]
        axs[0].bar(lon_x, lon_y, width=self.variogram.lon_res-0.1, align="edge")
        axs[0].set_xlabel(f"Lon [{self.units}]")
        axs[0].set_ylabel("Frequency")
        axs[0].set_xlim(left=0)

        lat_x = self.variogram.lat_res*(np.arange(self.variogram.lat_bins)[:view_range[1]])
        lat_y = np.sum(self.variogram.bins_count[:tol[0],:,:tol[2],0], axis=(0,2))[:view_range[1]]
        axs[1].bar(lat_x, lat_y, width=self.variogram.lat_res-0.1, align="edge")
        axs[1].set_xlabel(f"Lat [{self.units}]")
        axs[1].set_xlim(left=0)

        t_x = self.variogram.t_res*(np.arange(self.variogram.t_bins)[:view_range[2]])
        t_y = np.sum(self.variogram.bins_count[:tol[0],:tol[1],:,0], axis=(0,1))[:view_range[2]]
        axs[2].bar(t_x, t_y, width=self.variogram.t_res-0.1, align="edge")
        axs[2].set_xlabel("Time [hrs]")
        axs[2].set_xlim(left=0)
        plt.show()


    def plot_variograms(self, variable: AnyStr="u", tol: Tuple[int]=None, view_range:List[int]=None) -> None:
        """Plots the sliced variogram for each axis [lon, lat, time].
        
        tol -        the slicing tolerance when plotting over one dimension. If the variogram was
                     infinitely dense, ideally this would be one.
        view_range - three dim list which defines the index per axis used as a cut-off when
                     visualizing the sliced plots.
        """

        if self.variogram.bins is None:
            raise Exception("Need to run build_variogram() first!")
        if tol is None:
            tol = self.variogram.bins.shape[:3]
        if view_range is None:
            view_range = self.variogram.bins.shape[:3]
        
        variable_map = {"u": 0, "v": 1}
        try:
            var = variable_map[variable]
        except:
            raise ValueError("Specified variable does not exist")

        # (Note: division needed to normalize by dims of other bins)
        fig, axs = plt.subplots(1,3,figsize=(25,10))

        # Only divide if denom is non-zero, else zero (https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero)
        lon_y_num = np.sum(self.variogram.bins[:,:tol[1],:tol[2],var], axis=(1,2))[:view_range[0]]
        # lon_y_denom = self.variogram.bins.shape[1]*self.variogram.bins.shape[2]
        lon_y_denom = tol[1]*tol[2]
        axs[0].scatter(self.variogram.lon_res*(np.arange(self.variogram.lon_bins)[:view_range[0]]) + self.variogram.lon_res,\
            np.divide(lon_y_num, lon_y_denom, out=np.zeros_like(lon_y_num), where=lon_y_denom!=0), marker="x")
        axs[0].set_xlabel(f"Lon lag [{self.units}]")
        axs[0].set_ylabel("Semivariance")
        axs[0].set_xlim(left=0)
        axs[0].set_ylim([0,1.5])

        lat_y_num = np.sum(self.variogram.bins[:tol[0],:,:tol[2],var], axis=(0,2))[:view_range[1]]
        # lat_y_denom = self.variogram.bins.shape[0]*self.variogram.bins.shape[2]
        lat_y_denom = tol[0]*tol[2]
        axs[1].scatter(self.variogram.lat_res*(np.arange(self.variogram.lat_bins)[:view_range[1]]) + self.variogram.lat_res,\
            np.divide(lat_y_num, lat_y_denom, out=np.zeros_like(lat_y_num), where=lat_y_denom!=0), marker="x")
        axs[1].set_xlabel(f"Lat lag [{self.units}]")
        axs[1].set_xlim(left=0)
        axs[1].set_ylim([0,1.5])

        t_y_num = np.sum(self.variogram.bins[:tol[0],:tol[1],:,var], axis=(0,1))[:view_range[2]]
        # t_y_denom = self.variogram.bins.shape[0]*self.variogram.bins.shape[1]
        t_y_denom = tol[0]*tol[1]
        axs[2].scatter(self.variogram.t_res*(np.arange(self.variogram.t_bins)[:view_range[2]]) + self.variogram.t_res,\
            np.divide(t_y_num, t_y_denom, out=np.zeros_like(t_y_num), where=t_y_denom!=0), marker="x")
        axs[2].set_xlabel("Time lag [hrs]")
        axs[2].set_xlim(left=0)
        axs[2].set_ylim([0,1.5])
        plt.show() 


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
        var_y_num = np.sum(self.variogram.bins[:,:,:,var], axis=axis_map[axis_name]["other_axes"])[:cutoff]
        # var_y_denom = np.sum(self.variogram.bins_count[:,:,:,var], axis=axis_map[axis_name]["other_axes"])[:cutoff]
        var_y_denom = self.variogram.bins.shape[axis_map[axis_name]["other_axes"][0]]*self.variogram.bins.shape[axis_map[axis_name]["other_axes"][1]]
        axs[1].scatter(np.arange(axis_map[axis_name]["bins"])[:cutoff]*axis_map[axis_name]["res"], var_y_num/var_y_denom, marker="x")
        axs[1].title.set_text(f"{axis_name}")
        plt.show()


def load_variogram_from_npy(file_path: str):
    data = np.load(file_path, allow_pickle=True)
    bins = data.item().get("bins")
    bins_count = data.item().get("bins_count")
    res = data.item().get("res")
    units = data.item().get("units")
    return bins, bins_count, res, units
   