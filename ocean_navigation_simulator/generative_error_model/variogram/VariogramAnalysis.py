from ocean_navigation_simulator.generative_error_model.variogram.IndexPairGenerator import IndexPairGenerator

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, AnyStr
import matplotlib.pyplot as plt
import itertools

# TODO: if normal variogram method does not work due to RAM constraints, automatically
# call the generator-based method

class VariogramAnalysis:
    """
    Handles all the required variogram analysis functionality needed.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.bins = None
        self.bins_count = None


    def detrend(self, detrend_var: AnyStr="lat", num_bins: int=5) -> None:
        """Variogram analysis assumes stationarity, therefore detrending is needed. This method
        can perform detrending over a specific variable (e.g. lon, lat, time) by subtracting the 
        mean and dividing by standard deviation"""

        min_val = np.floor(np.min(self.data[detrend_var].to_numpy()))
        max_val = np.ceil(np.max(self.data[detrend_var].to_numpy()))
        bin_step = (max_val-min_val)/num_bins
        bins = np.arange(min_val, max_val+bin_step, bin_step)
        bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
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
        self.detrend_var, self.bin_labels = detrend_var, bin_labels

        # subtract mean and divide by std with bin statistics
        self.data["detrended_u_error"] = self.data.apply(lambda x: (x["u_error"] - bin_statistics[x[f"{detrend_var}_bins"]]["u_error"][0])\
            /bin_statistics[x[f"{detrend_var}_bins"]]["u_error"][1], axis=1)
        self.data["detrended_v_error"] = self.data.apply(lambda x: (x["v_error"] - bin_statistics[x[f"{detrend_var}_bins"]]["v_error"][0]\
            )/bin_statistics[x[f"{detrend_var}_bins"]]["v_error"][1], axis=1)


    def build_variogram(self, res_tuple: Tuple[int], total_res: int, chunk_size:int, cross_buoy_pairs_only: bool=True,\
        detrended: bool=False) -> List[np.ndarray]:

        """Find all possible pairs of points. It then computes the lag value in each axis
        and the variogram value for u and v errors.
        
        res_tuple - resolution in (degrees, degrees, hours)
        total_res - resolution if variogram is 1D
        bins_tuple - number of bins in (lon, lat, time)
        chunk_size - lower number if less memory
        cross_buoy_pairs_only - whether or not to use point pairs from the same buoy
        detrended - use detrended value or not
        """

        if detrended and "detrended_u_error" not in self.data.columns:
            raise Exception("Need to run detrend method first with 'detrend=True'")

        # convert time axis to datetime
        self.data["time"] = pd.to_datetime(self.data["time"])
        # find earliest time/date and subtract from time column
        earliest_date = self.data["time"].min()
        self.data["time_offset"] = self.data["time"].apply(lambda x: (x - earliest_date).seconds//3600 + (x-earliest_date).days*24)

        n = self.data.shape[0]
        self.lon_res, self.lat_res, self.t_res = lon_res, lat_res, t_res = res_tuple
        self.total_res = total_res

        # create bins from known extremal values and resolutions
        self.t_bins = np.ceil((self.data["time_offset"].max() - self.data["time_offset"].min()+1)/t_res).astype(int)
        self.lon_bins = np.ceil((self.data["lon"].max() - self.data["lon"].min()+1)/lon_res).astype(int)
        self.lat_bins = np.ceil((self.data["lat"].max() - self.data["lat"].min()+1)/lat_res).astype(int)
        bin_sizes = (self.lon_bins, self.lat_bins, self.t_bins)

        # get indices of all pairs
        i,j = np.triu_indices(n, k=1)

        # build a mask to mask out the pairs from same buoy
        if cross_buoy_pairs_only:
            buoy_vector = self.data["buoy"].to_numpy()
            # map each buoy name string to unique integer
            _, integer_mapped = np.unique(buoy_vector, return_inverse=True)
            i_temp, j_temp = np.triu_indices(len(integer_mapped), k=1)
            # if integer in i and j is the same, then both points from same buoy
            residual = (integer_mapped[i_temp] - integer_mapped[j_temp])
            mask_same_buoy = residual != 0
            mask_idx = np.where(mask_same_buoy == True)

            # only select pairs from different buoys
            i = i[mask_idx]
            j = j[mask_idx]
        print(f"Number of pairs of points: {len(i)}")

        # make bins
        bins = np.zeros((*bin_sizes, 2))
        bins_count = np.zeros_like(bins)

        # make total bins
        total_bin_num = np.ceil((np.sqrt(self.t_bins**2 + self.lon_bins**2 + self.lat_bins**2)/total_res)+1).astype(int)
        total_bins = np.zeros((total_bin_num,2))
        total_bins_count = np.zeros_like(total_bins)

        # convert relevant columns to numpy before accessing
        time = self.data["time_offset"].to_numpy()
        lon = self.data["lon"].to_numpy()
        lat = self.data["lat"].to_numpy()
        if detrended:
            u_error = self.data["detrended_u_error"].to_numpy()
            v_error = self.data["detrended_v_error"].to_numpy()
        else:
            u_error = self.data["u_error"].to_numpy()
            v_error = self.data["v_error"].to_numpy()

        offset = 0
        for chunk in tqdm(range(chunk_size, len(i)+1, chunk_size)):
            # get values for chunk
            idx_i = i[0+offset:chunk]
            idx_j = j[0+offset:chunk]

            # get lags and divide by bin resolution to get bin index
            t_lag = np.floor((np.absolute(time[idx_i] - time[idx_j])/t_res).astype(float)).astype(int)
            lon_lag = np.floor(np.absolute(lon[idx_i] - lon[idx_j])/lon_res).astype(int)
            lat_lag = np.floor(np.absolute(lat[idx_i] - lat[idx_j])/lat_res).astype(int)
            u_squared_diff = np.square(u_error[idx_i] - u_error[idx_j]).reshape(-1,1)
            v_squared_diff = np.square(v_error[idx_i] - v_error[idx_j]).reshape(-1,1)
            squared_diff = np.hstack((u_squared_diff, v_squared_diff))

            # total lag vector
            total_lag = np.floor(np.sqrt(np.square((time[idx_i] - time[idx_j]).astype(float)) + \
                np.square(lon[idx_i] - lon[idx_j]) + np.square(lat[idx_i] - lat[idx_j]))/total_res).astype(int)

            # assign values to bin
            if np.sum(bins) == 0:
                # from: https://stackoverflow.com/questions/51092737/vectorized-assignment-in-numpy
                # add to relevant bin
                np.add.at(bins, (lon_lag, lat_lag, t_lag), squared_diff)
                np.add.at(total_bins, total_lag, squared_diff)
                # add to bin count
                np.add.at(bins_count, (lon_lag, lat_lag, t_lag), [1,1])
                np.add.at(total_bins_count, total_lag, [1,1])
            else:
                bins_temp = np.zeros_like(bins)
                bins_count_temp = np.zeros_like(bins_temp)
                total_bins_temp = np.zeros_like(total_bins)
                total_bins_count_temp = np.zeros_like(total_bins_temp)
                # perform operations
                np.add.at(bins_temp, (lon_lag, lat_lag, t_lag), squared_diff)
                np.add.at(bins_count_temp, (lon_lag, lat_lag, t_lag), [1,1])
                np.add.at(total_bins_temp, total_lag, squared_diff)
                np.add.at(total_bins_count_temp, total_lag, [1,1])
                bins += bins_temp
                bins_count += bins_count_temp
                total_bins += total_bins_temp
                total_bins_count += total_bins_count_temp

            offset += chunk_size
        bins = np.divide(bins, bins_count, out=np.zeros_like(bins), where=bins_count!=0)/2
        total_bins = np.divide(total_bins, total_bins_count, out=np.zeros_like(total_bins), where=total_bins_count!=0)/2
        self.bins, self.bins_count = bins, bins_count
        self.total_bins, self.total_bins_count = total_bins, total_bins_count
        return bins, bins_count


    def build_variogram_gen(self, res_tuple: Tuple[int], total_res: int, chunk_size:int, cross_buoy_pairs_only: bool=True,\
        detrended: bool=False) -> List[np.ndarray]:

        """Find all possible pairs of points. It then computes the lag value in each axis
        and the variogram value for u and v errors.

        This method uses a generator if there are too many index pairs to hold in RAM.
        
        res_tuple - resolution in (degrees, degrees, hours)
        bins_tuple - number of bins in (lon, lat, time)
        chunk_size - lower number if less memory
        cross_buoy_pairs_only - whether or not to use point pairs from the same buoy
        detrended - use detrended value or not
        """

        if detrended and "detrended_u_error" not in self.data.columns:
            raise Exception("Need to run detrend method first with 'detrend=True'")

        # convert time axis to datetime
        self.data["time"] = pd.to_datetime(self.data["time"])
        # find earliest time/date and subtract from time column
        earliest_date = self.data["time"].min()
        self.data["time_offset"] = self.data["time"].apply(lambda x: (x - earliest_date).seconds//3600 + (x-earliest_date).days*24)

        self.lon_res, self.lat_res, self.t_res = lon_res, lat_res, t_res = res_tuple
        self.total_res = total_res

        # create bins from known extremal values and resolutions
        self.t_bins = np.ceil((self.data["time_offset"].max() - self.data["time_offset"].min()+1)/t_res).astype(int)
        self.lon_bins = np.ceil((self.data["lon"].max() - self.data["lon"].min()+1)/lon_res).astype(int)
        self.lat_bins = np.ceil((self.data["lat"].max() - self.data["lat"].min()+1)/lat_res).astype(int)
        bin_sizes = (self.lon_bins, self.lat_bins, self.t_bins)

        # setup generator to generate index pairs
        n = self.data.shape[0]
        gen = IndexPairGenerator(n, chunk_size)

        # build a mask to mask out the pairs from same buoy
        buoy_vector = None
        if cross_buoy_pairs_only:
            buoy_vector = self.data["buoy"].to_numpy()

        # make bins
        self.bins = np.zeros((*bin_sizes,2))
        self.bins_count = np.zeros_like(self.bins)

        # make total bins
        total_bin_num = np.ceil((np.sqrt(self.t_bins**2 + self.lon_bins**2 + self.lat_bins**2)/total_res)+1).astype(int)
        self.total_bins = np.zeros((total_bin_num,2))
        self.total_bins_count = np.zeros_like(self.total_bins)

        # convert relevant columns to numpy before accessing
        time = self.data["time_offset"].to_numpy()
        lon = self.data["lon"].to_numpy()
        lat = self.data["lat"].to_numpy()
        if detrended:
            u_error = self.data["detrended_u_error"].to_numpy()
            v_error = self.data["detrended_v_error"].to_numpy()
        else:
            u_error = self.data["u_error"].to_numpy()
            v_error = self.data["v_error"].to_numpy()

        # iterate over generator to get relevant indices
        while True:
            indices = next(gen)
            if len(indices[0]) == 0:
                break
            self._calculate_chunk(indices, time, lon, lat, u_error, v_error, buoy_vector)
        
        # divide bins by the count + divide by 2 for semi-variance (special func to avoid dividing by zero)
        self.bins = np.divide(self.bins, self.bins_count, out=np.zeros_like(self.bins), where=self.bins_count!=0)/2
        self.total_bins = np.divide(self.total_bins, self.total_bins_count, out=np.zeros_like(self.total_bins), where=self.total_bins_count!=0)/2

        return self.bins, self.bins_count


    def _calculate_chunk(self, indices: List[int], time: List[int], lon: List[int], lat: List[int],\
        u_error: List[int], v_error: List[int], buoy_vector: List[str]) -> None:
        # get values for chunk
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

        # get lags and divide by bin resolution to get bin index
        t_lag = np.floor((np.absolute(time[idx_i] - time[idx_j])/self.t_res).astype(float)).astype(int)
        lon_lag = np.floor(np.absolute(lon[idx_i] - lon[idx_j])/self.lon_res).astype(int)
        lat_lag = np.floor(np.absolute(lat[idx_i] - lat[idx_j])/self.lat_res).astype(int)
        u_squared_diff = np.square(u_error[idx_i] - u_error[idx_j]).reshape(-1,1)
        v_squared_diff = np.square(v_error[idx_i] - v_error[idx_j]).reshape(-1,1)
        squared_diff = np.hstack((u_squared_diff, v_squared_diff))

        # total lag vector
        total_lag = np.floor(np.sqrt(np.square((time[idx_i] - time[idx_j]).astype(float)) + \
            np.square(lon[idx_i] - lon[idx_j]) + np.square(lat[idx_i] - lat[idx_j]))/self.total_res).astype(int)

        # assign values to bin
        if np.sum(self.bins) == 0:
            # from: https://stackoverflow.com/questions/51092737/vectorized-assignment-in-numpy
            # add to relevant bin
            np.add.at(self.bins, (lon_lag, lat_lag, t_lag), squared_diff)
            np.add.at(self.total_bins, total_lag, squared_diff)

            # add to bin count
            np.add.at(self.bins_count, (lon_lag, lat_lag, t_lag), [1,1])
            np.add.at(self.total_bins_count, total_lag, [1,1])
        else:
            bins_temp = np.zeros_like(self.bins)
            bins_count_temp = np.zeros_like(self.bins_count)
            total_bins_temp = np.zeros_like(self.total_bins)
            total_bins_count_temp = np.zeros_like(total_bins_temp)
            # perform operations
            np.add.at(bins_temp, (lon_lag, lat_lag, t_lag), squared_diff)
            np.add.at(bins_count_temp, (lon_lag, lat_lag, t_lag), [1,1])
            np.add.at(total_bins_temp, total_lag, squared_diff)
            np.add.at(total_bins_count_temp, total_lag, [1,1])
            self.bins += bins_temp
            self.bins_count += bins_count_temp
            self.total_bins += total_bins_temp
            self.total_bins_count += total_bins_count_temp

    
    def plot_detrended_bins(self) -> None:
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


    def plot_histograms(self) -> None:
        if self.bins is None:
            raise Exception("Need to run build_variogram() first!")

        # plot histogram for each axis
        fig, axs = plt.subplots(1,3,figsize=(25,10))
        axs[0].bar(np.arange(self.lon_bins)*self.lon_res, np.sum(self.bins_count[:,:,:,0], axis=(1,2)))
        axs[0].set_xlabel("Lon [degrees]")
        axs[0].set_ylabel("Frequency")
        axs[1].bar(np.arange(self.lat_bins)*self.lat_res, np.sum(self.bins_count[:,:,:,0], axis=(0,2)))
        axs[1].set_xlabel("Lat [degrees]")
        axs[2].bar(np.arange(self.t_bins)*self.t_res, np.sum(self.bins_count[:,:,:,0], axis=(0,1)))
        axs[2].set_xlabel("Time [hrs]")
        plt.show()

    def plot_hist_vario_for_axis(self, axis_name: str="time", variable: str="u", cutoff=5000) -> None:
        if self.bins is None:
            raise Exception("Need to run build_variogram() first!")

        variable_map = {"u": 0, "v": 1}
        try:
            var = variable_map[variable]
        except:
            raise ValueError("Specified variable does not exist")

        axis_map = {"lon": {"bins":self.lon_bins, "res":self.lon_res, "other_axes":(1,2)},
                    "lat": {"bins":self.lat_bins, "res":self.lat_res, "other_axes":(0,2)},
                    "time": {"bins":self.t_bins, "res":self.t_res, "other_axes":(0,1)}}

        fig, axs = plt.subplots(2,1,figsize=(20,20))
        axs[0].bar(np.arange(axis_map[axis_name]["bins"])[:cutoff]*axis_map[axis_name]["res"], np.sum(self.bins_count[:,:,:,0], axis=axis_map[axis_name]["other_axes"])[:cutoff])
        axs[0].title.set_text(f"{axis_name}")
        var_y_num = np.sum(self.bins[:,:,:,var], axis=axis_map[axis_name]["other_axes"])[:cutoff]
        # var_y_denom = np.sum(self.bins_count[:,:,:,var], axis=axis_map[axis_name]["other_axes"])[:cutoff]
        var_y_denom = self.bins.shape[axis_map[axis_name]["other_axes"][0]]*self.bins.shape[axis_map[axis_name]["other_axes"][1]]
        axs[1].scatter(np.arange(axis_map[axis_name]["bins"])[:cutoff]*axis_map[axis_name]["res"], var_y_num/var_y_denom, marker="x")
        axs[1].title.set_text(f"{axis_name}")
        plt.show()


    def plot_variograms(self, variable: AnyStr="u") -> None:
        if self.bins is None:
            raise Exception("Need to run build_variogram() first!")

        variable_map = {"u": 0, "v": 1}

        try:
            var = variable_map[variable]
        except:
            raise ValueError("Specified variable does not exist")

        # plot variograms for either u or v
        # (Note: division needed to normalize by dims of other bins)
        fig, axs = plt.subplots(1,3,figsize=(25,10))

        # Only divide if denom is non-zero, else zero (https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero)
        lon_y_num = np.sum(self.bins[:,:,:,var], axis=(1,2))
        lon_y_denom = self.bins.shape[1]*self.bins.shape[2]
        axs[0].scatter(np.arange(self.lon_bins)*self.lon_res, np.divide(lon_y_num, lon_y_denom, \
            out=np.zeros_like(lon_y_num), where=lon_y_denom!=0), marker="x")
        axs[0].set_xlabel("Lon lag [degrees]")
        axs[0].set_ylabel("Semivariance")

        lat_y_num = np.sum(self.bins[:,:,:,var], axis=(0,2))
        lat_y_denom = self.bins.shape[0]*self.bins.shape[2]
        axs[1].scatter(np.arange(self.lat_bins)*self.lat_res, np.divide(lat_y_num, lat_y_denom, \
            out=np.zeros_like(lat_y_num), where=lat_y_denom!=0), marker="x")
        axs[1].set_xlabel("Lat lag [degrees]")

        t_y_num = np.sum(self.bins[:,:,:,var], axis=(0,1))
        t_y_denom = self.bins.shape[0]*self.bins.shape[1]
        axs[2].scatter(np.arange(self.t_bins)*self.t_res, np.divide(t_y_num, t_y_denom, \
            out=np.zeros_like(t_y_num), where=t_y_denom!=0), marker="x")
        axs[2].set_xlabel("Time lag [hrs]")
        plt.show()