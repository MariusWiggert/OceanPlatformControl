from ocean_navigation_simulator.generative_error_model.variogram.IndexPairGenerator import IndexPairGenerator

import pandas as pd
import numpy as np
from typing import Tuple, List, AnyStr, Dict
import matplotlib.pyplot as plt
import multiprocessing as mp
import ctypes as c
import logging


class Variogram:
    """
    Handles all the required variogram analysis functionality needed.
    """

    def __init__(self, data: pd.DataFrame=None):
        self.data = data
        self.bins = None
        self.bins_count = None

    def detrend(self, detrend_var: AnyStr="lat", num_bins: int=5) -> Dict[str, List[float]]:
        """Variogram analysis assumes stationarity, therefore detrending is needed.
        
        This method can perform detrending over a specific variable (e.g. lon, lat, time)
        by subtracting the mean and dividing by standard deviation."""

        if detrend_var == "time":
            # convert time axis to datetime
            self.data["time"] = pd.to_datetime(self.data["time"])
            # find earliest time/date and subtract from time column
            earliest_date = self.data["time"].min()
            self.data["time_offset"] = self.data["time"].apply(lambda x: (x - earliest_date).seconds//3600 + (x-earliest_date).days*24)
            detrend_var = "time_offset"

        min_val = np.floor(np.min(self.data[detrend_var].to_numpy()))
        max_val = np.ceil(np.max(self.data[detrend_var].to_numpy()))
        bin_step = (max_val-min_val)/num_bins
        bins = np.arange(min_val, max_val+bin_step, bin_step)
        bin_labels = [f"{bins[i]},{bins[i+1]}" for i in range(len(bins)-1)]

        # create new col with bin name for each row
        self.data[f"{detrend_var}_bins"] = pd.cut(self.data[detrend_var], bins, labels=bin_labels, include_lowest=True)

        bin_statistics = {}
        for bin_label in bin_labels:
            u_stats = []
            u_stats.append(self.data[self.data[f"{detrend_var}_bins"] == bin_label]["u_error"].mean())
            u_stats.append(np.sqrt(self.data[self.data[f"{detrend_var}_bins"] == bin_label]["u_error"].var()))

            v_stats = []
            v_stats.append(self.data[self.data[f"{detrend_var}_bins"] == bin_label]["v_error"].mean())
            v_stats.append(np.sqrt(self.data[self.data[f"{detrend_var}_bins"] == bin_label]["v_error"].var()))

            bin_statistics[bin_label] = {"u_error": u_stats, "v_error": v_stats}

        # make accessible for plotting function
        self.detrend_var, self.bin_labels, self.bin_statistics = detrend_var, bin_labels, bin_statistics

        # subtract mean and divide by std with bin statistics
        self.data["detrended_u_error"] = self.data.apply(lambda x: (x["u_error"] - bin_statistics[x[f"{detrend_var}_bins"]]["u_error"][0])\
            /bin_statistics[x[f"{detrend_var}_bins"]]["u_error"][1], axis=1)
        self.data["detrended_v_error"] = self.data.apply(lambda x: (x["v_error"] - bin_statistics[x[f"{detrend_var}_bins"]]["v_error"][0]\
            )/bin_statistics[x[f"{detrend_var}_bins"]]["v_error"][1], axis=1)
        return bin_statistics

    def build_variogram_gen(self, res_tuple: Tuple[float], num_workers: int, chunk_size:int, cross_buoy_pairs_only: bool=True,\
        detrended: bool=False, units: str="km", logger: logging=None) -> Tuple[np.ndarray, np.ndarray]:

        """Find all possible pairs of points. Then computes the lag value in each axis
        and the variogram value for u and v errors.

        This method uses a generator if there are too many index pairs to hold in RAM.
        
        res_tuple - resolution in (degrees, degrees, hours)
        num_workers - workers for multi-processing
        bins_tuple - number of bins in (lon, lat, time)
        chunk_size - lower number if less memory
        cross_buoy_pairs_only - whether or not to use point pairs from the same buoy
        detrended - use detrended value or not
        """

        if detrended and "detrended_u_error" not in self.data.columns:
            raise Exception("Need to run detrend method first with 'detrend=True'")

        self.units = units
        self._variogram_setup(res_tuple)

        # setup generator to generate index pairs
        n = self.data.shape[0]
        gen = IndexPairGenerator(n, chunk_size)

        # make bins
        bin_sizes = (self.lon_bins, self.lat_bins, self.t_bins)
        self.bins = np.zeros((*bin_sizes,2), dtype=np.float32)
        self.bins_count = np.zeros_like(self.bins, dtype=np.int32)

        # convert relevant columns to numpy before accessing
        time = self.data["time_offset"].to_numpy(dtype=np.float32)
        lon = self.data["lon"].to_numpy(dtype=np.float32)
        lat = self.data["lat"].to_numpy(dtype=np.float32)
        if detrended:
            u_error = self.data["detrended_u_error"].to_numpy(dtype=np.float32)
            v_error = self.data["detrended_v_error"].to_numpy(dtype=np.float32)
        else:
            u_error = self.data["u_error"].to_numpy(dtype=np.float32)
            v_error = self.data["v_error"].to_numpy(dtype=np.float32)
        buoy_vector = None
        if cross_buoy_pairs_only:
            buoy_vector = self.data["buoy"].to_numpy()

        # setup shared arrays for workers
        shared_bins = to_shared_array(self.bins, c.c_float)
        self.bins = to_numpy_array(shared_bins, self.bins.shape)

        shared_bins_count = to_shared_array(self.bins_count, c.c_int32)
        self.bins_count = to_numpy_array(shared_bins_count, self.bins_count.shape)

        # setup mp stuff
        q = mp.Queue(maxsize=num_workers)
        iolock = mp.Lock()
        pool = mp.Pool(num_workers, initializer=self._calculate_chunk,\
             initargs=(q, time, lon, lat, u_error, v_error, buoy_vector, iolock))

        # iterate over generator to get relevant indices
        number_of_pairs = (n**2)/2 - n
        running_sum = 0
        iteration = 0
        while True:
            indices = next(gen)
            if len(indices[0]) == 0:
                break
            q.put(indices)
            with iolock:
                running_sum += len(indices[0])
                if iteration % 10 == 0 and iteration != 0 and logger is not None:
                    logger.info(f"Iteration: {iteration}, Estimated {round(100*(running_sum/number_of_pairs),2)}% of pairs finished.")
                iteration += 1

        for _ in range(num_workers):
            q.put(None)

        pool.close()
        pool.join()
        
        # divide bins by the count + divide by 2 for semi-variance (special func to avoid dividing by zero)
        self.bins = np.divide(self.bins, self.bins_count, out=np.zeros_like(self.bins), where=self.bins_count!=0)/2
        return self.bins, self.bins_count

    def _calculate_chunk(self, q: mp.Queue, time: List[int], lon: List[int], lat: List[int],\
        u_error: List[int], v_error: List[int], buoy_vector: List[str], iolock: mp.Lock) -> None:
        """Used in multiprocessing to compute the variogram values of pairs of points according to
        the indices."""

        while True:
            # get values for chunk
            indices = q.get()
            if indices is None:
                break
            idx_i = np.array(indices[0])
            idx_j = np.array(indices[1])

            # build mask to eliminate pairs from same buoy
            if buoy_vector is not None:
                buoy_vector_i = buoy_vector[idx_i]
                buoy_vector_j = buoy_vector[idx_j]
                # map each buoy name string to unique integer
                _, integer_mapped = np.unique([buoy_vector_i, buoy_vector_j], return_inverse=True)
                buoy_vector_mapped_i = integer_mapped[:round(len(integer_mapped)/2)]
                buoy_vector_mapped_j = integer_mapped[round(len(integer_mapped)/2):]
                # if integer in i and j is the same, then both points from same buoy
                residual = buoy_vector_mapped_i - buoy_vector_mapped_j
                mask_same_buoy = residual != 0
                mask_idx = np.where(mask_same_buoy == True)

                # only select pairs from different buoys
                idx_i = idx_i[mask_idx]
                idx_j = idx_j[mask_idx]

            # get lags in degrees and divide by bin resolution to get bin indices
            if self.units == "degrees":
                lon_lag = np.floor(np.absolute(lon[idx_i] - lon[idx_j])/self.lon_res).astype(int)
                lat_lag = np.floor(np.absolute(lat[idx_i] - lat[idx_j])/self.lat_res).astype(int)
            elif self.units == "km":
            # convert lags from degrees to kilometres
                pts1 = np.hstack((lon[idx_i].reshape(-1, 1), lat[idx_i].reshape(-1,1)))
                pts2 = np.hstack((lon[idx_j].reshape(-1, 1), lat[idx_j].reshape(-1,1)))
                lon_lag, lat_lag = convert_degree_to_km(pts1, pts2)
                # convert to bin indices
                lon_lag = np.floor(np.absolute(lon_lag)/self.lon_res).astype(int)
                lat_lag = np.floor(np.absolute(lat_lag)/self.lat_res).astype(int)

            t_lag = np.floor((np.absolute(time[idx_i] - time[idx_j])/self.t_res).astype(float)).astype(int)

            u_squared_diff = np.square(u_error[idx_i] - u_error[idx_j]).reshape(-1,1)
            v_squared_diff = np.square(v_error[idx_i] - v_error[idx_j]).reshape(-1,1)
            squared_diff = np.hstack((u_squared_diff, v_squared_diff))

            # assign values to bin
            if np.sum(self.bins) == 0:
                # from: https://stackoverflow.com/questions/51092737/vectorized-assignment-in-numpy
                with iolock:
                    # add to relevant bin
                    np.add.at(self.bins, (lon_lag, lat_lag, t_lag), squared_diff)
                    # add to bin count
                    np.add.at(self.bins_count, (lon_lag, lat_lag, t_lag), [1,1])
            else:
                bins_temp = np.zeros_like(self.bins)
                bins_count_temp = np.zeros_like(self.bins_count)
                # perform operations
                np.add.at(bins_temp, (lon_lag, lat_lag, t_lag), squared_diff)
                np.add.at(bins_count_temp, (lon_lag, lat_lag, t_lag), [1,1])
                with iolock:
                    self.bins += bins_temp
                    self.bins_count += bins_count_temp

    def _variogram_setup(self, res_tuple: Tuple[float]):
        """Helper method to declutter the main build_variogram method."""

        # convert time axis to datetime
        self.data["time"] = pd.to_datetime(self.data["time"])
        # find earliest time/date and subtract from time column
        earliest_date = self.data["time"].min()
        self.data["time_offset"] = self.data["time"].apply(lambda x: (x - earliest_date).seconds//3600 + (x-earliest_date).days*24)

        self.lon_res, self.lat_res, self.t_res = lon_res, lat_res, t_res = res_tuple

        # calculate bin sizes from known extremal values and given resolutions
        self.t_bins = np.ceil((self.data["time_offset"].max() - self.data["time_offset"].min()+1)/t_res).astype(int)
        if self.units == "degrees":
            self.lon_bins = np.ceil((self.data["lon"].max() - self.data["lon"].min()+1)/lon_res).astype(int)
            self.lat_bins = np.ceil((self.data["lat"].max() - self.data["lat"].min()+1)/lat_res).astype(int)
        elif self.units == "km":
            max_lon, min_lon = self.data["lon"].max(), self.data["lon"].min()
            max_lat, min_lat = self.data["lat"].max(), self.data["lat"].min()
            max_dx, max_dy = convert_degree_to_km(np.array([min_lon, min_lat]), np.array([max_lon, max_lat]))
            self.lon_bins = np.ceil(abs(max_dx/lon_res)+5).astype(int)
            self.lat_bins = np.ceil(abs(max_dy/lat_res)+5).astype(int)

    def plot_detrended_bins(self) -> None:
        """Plots detrended data over time for each bin separately."""

        fig, axs = plt.subplots(len(self.bin_labels),1,figsize=(20,len(self.bin_labels)*6))

        for i, bin_label in enumerate(self.bin_labels):
            detrended_u_error = self.data[self.data[f"{self.detrend_var}_bins"] == bin_label]["detrended_u_error"].to_numpy()
            detrended_v_error = self.data[self.data[f"{self.detrend_var}_bins"] == bin_label]["detrended_v_error"].to_numpy()
            detrended_RMSE = np.sqrt(detrended_u_error**2 + detrended_v_error**2)
            bin_time = self.data[self.data[f"{self.detrend_var}_bins"] == bin_label]["time"]
            df_temp = pd.DataFrame({"time": bin_time, "detrended_RMSE": detrended_RMSE, "u_error_detrended": detrended_u_error,\
                                    "v_error_detrended": detrended_v_error})
            detrended_group = df_temp.groupby(by=["time"], as_index=False).mean()
            if len(self.bin_labels) == 1:
                axs.plot(detrended_group["time"], detrended_group["u_error_detrended"], label="u error binned and detrended")
                axs.plot(detrended_group["time"], detrended_group["v_error_detrended"], label="v error binned and detrended")
                axs.title.set_text(f"Plot for '{self.detrend_var}' range of {bin_label} degrees (no binning)")
                axs.set_xticks(np.arange(0, len(detrended_group), round(len(detrended_group)/8)))
                axs.grid()
            else:
                axs[i].plot(detrended_group["time"], detrended_group["u_error_detrended"], label="u error binned and detrended")
                axs[i].plot(detrended_group["time"], detrended_group["v_error_detrended"], label="v error binned and detrended")
                axs[i].title.set_text(f"Plot for '{self.detrend_var}' range of {bin_label} degrees")
                axs[i].set_xticks(np.arange(0, len(detrended_group), round(len(detrended_group)/8)))
                axs[i].grid()
        plt.legend()
        plt.show()

    def plot_bin_stats(self, detrend_var: str="lat", num_bins=5) -> None:
        """Detrends over specified variable and plots the bin statistics over that variable."""

        bin_dict = self.detrend(detrend_var, num_bins)

        length = len(bin_dict.keys())
        mean_u = np.empty(length)
        std_u = np.empty(length)
        mean_v = np.empty(length)
        std_v = np.empty(length)
        x = np.empty(length)

        for i, key in enumerate(bin_dict.keys()):
            x[i] = float(key.split(",")[-1])
            mean_u[i] = (bin_dict[key]["u_error"][0])
            std_u[i] = (bin_dict[key]["u_error"][1])
            mean_v[i] = (bin_dict[key]["v_error"][0])
            std_v[i] = (bin_dict[key]["v_error"][1])

        fig, axs = plt.subplots(1,2, figsize=(20,8))
        axs[0].plot(x, mean_u, label="mean u")
        axs[0].fill_between(x, mean_u-2*std_u, mean_u+2*std_u, facecolor="b", alpha=0.5)
        axs[0].title.set_text("u error mean +- 2 std")
        axs[0].set_ylim([-1,1])
        axs[0].grid()
        axs[1].plot(x, mean_v, label="mean v")
        axs[1].fill_between(x, mean_v-2*std_v, mean_v+2*std_v, facecolor="b", alpha=0.5)
        axs[1].title.set_text("v error mean +- 2 std")
        axs[1].set_ylim([-1,1])
        axs[1].grid()

        fig.suptitle(f"Trend for '{detrend_var}'", size=20)
        plt.show()


#------------------------ Helper Funcs -----------------------#


def to_shared_array(arr, ctype):
    shared_array = mp.Array(ctype, arr.size, lock=False)
    temp = np.frombuffer(shared_array, dtype=arr.dtype)
    temp[:] = arr.flatten(order='C')
    return shared_array


def to_numpy_array(shared_array, shape):
    """Create a numpy array backed by a shared memory Array."""
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)


def create_shared_array_from_np(arr, ctype):
    """Combines two functions, implemented due to repetition"""
    shared_array = to_shared_array(arr, ctype)
    output = to_numpy_array(shared_array, arr.shape)
    return output


def convert_degree_to_km(pts1: np.ndarray, pts2: np.ndarray) -> List[np.ndarray]:
    """Takes two sets of points, each with a lat and lon degree, and computes the distance between each pair in km.
    Note: e.g. pts1 -> np.array([lon, lat])."""
    # https://stackoverflow.com/questions/24617013/convert-latitude-and-longitude-to-x-and-y-grid-system-using-python
    if len(pts1.shape) > 1:
        dx = (pts1[:, 0] - pts2[:, 0]) * 40075.2 * np.cos((pts1[:, 1] + pts2[:, 1]) * np.pi/360)/360
        dy = ((pts1[:, 1] - pts2[:, 1]) * 39806.64)/360
    else:
        dx = (pts1[0] - pts2[0]) * 40075.2 * np.cos((pts1[1] + pts2[1]) * np.pi/360)/360
        dy = ((pts1[1] - pts2[1]) * 39806.64)/360
    return dx, dy
