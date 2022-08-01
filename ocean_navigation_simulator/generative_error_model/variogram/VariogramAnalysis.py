import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, AnyStr
import matplotlib.pyplot as plt
import itertools

class VariogramAnalysis:
    """
    Handles all the required variogram analysis functionality needed.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.bins = None
        self.bins_count = None

    def detrend(self, axis: AnyStr="lat"):
        """Variogram analysis assumes stationarity, therefore detrending is needed. This method
        can perform detrending over a specific axis by subtracting the mean and dividing by
        standard deviation"""

        pass

    def build_variogram(self, res_tuple: Tuple[int],\
        bins_tuple: Tuple[int], chunk_size:int) -> List[np.ndarray]:
        """Find all possible pairs of points. It then computes the lag value in each axis
        and the variogram value for u and v errors."""

        n = self.data.shape[0]
        self.lon_res, self.lat_res, self.t_res = lon_res, lat_res, t_res = res_tuple
        self.lon_bins, self.lat_bins, self.t_bins = lon_bins, lat_bins, t_bins = bins_tuple

        # get indices of all pairs
        i,j = np.triu_indices(n, k=1)

        # build a mask to mask out the pairs from same buoy
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

        # convert time axis to datetime
        self.data["time"] = pd.to_datetime(self.data["time"])

        # make bins
        bins = np.zeros((lon_bins,lat_bins,t_bins,2)) # 5 km, 5km, 6hrs steps
        bins_count = np.zeros((lon_bins, lat_bins, t_bins,2))

        # convert relevant to numpy before accessing
        time = self.data["time"].to_numpy()
        lon = self.data["lon"].to_numpy()
        lat = self.data["lat"].to_numpy()
        u_error = self.data["u_error"].to_numpy()
        v_error = self.data["v_error"].to_numpy()

        offset = 0
        for chunk in tqdm(range(chunk_size, len(i), chunk_size)):
            # get values for chunk
            idx_i = i[0+offset:chunk]
            idx_j = j[0+offset:chunk]

            # get lags
            t_lag = np.floor((np.absolute((time[idx_i] - time[idx_j])/3.6e12)/t_res).astype(float)).astype(int)
            lon_lag = np.floor(np.absolute(lon[idx_i] - lon[idx_j])/lon_res).astype(int)
            lat_lag = np.floor(np.absolute(lat[idx_i] - lat[idx_j])/lat_res).astype(int)
            u_squared_diff = np.square(u_error[idx_i] - u_error[idx_j])
            v_squared_diff = np.square(v_error[idx_i] - v_error[idx_j])
            squared_diff = np.array(np.vstack((u_squared_diff, v_squared_diff)).reshape((-1,2)))

            # assign values to bin
            if np.sum(bins) == 0:
                # from: https://stackoverflow.com/questions/51092737/vectorized-assignment-in-numpy
                # add to relevant bin
                np.add.at(bins, (lon_lag, lat_lag, t_lag), squared_diff)
                # add to bin count
                np.add.at(bins_count, (lon_lag, lat_lag, t_lag), [1,1])
            else:
                bins_temp = np.zeros((lon_bins,lat_bins,t_bins,2))
                bins_count_temp = np.zeros(bins_temp.shape)
                # perform operations
                np.add.at(bins_temp, (lon_lag, lat_lag, t_lag), squared_diff)
                np.add.at(bins_count_temp, (lon_lag, lat_lag, t_lag), [1,1])
                bins += bins_temp
                bins_count += bins_count_temp

            offset += chunk_size
        
        bins[lon_lag, lat_lag, t_lag] /= bins_count[lon_lag, lat_lag, t_lag]
        self.bins = bins
        self.bins_count = bins_count
        return bins, bins_count


    def plot_histograms(self) -> None:
        if self.bins is None:
            raise Exception("Need to run build_variogram() first!")

        # plot histogram for each axis
        fig, axs = plt.subplots(1,3,figsize=(25,10))
        axs[0].bar(np.arange(self.t_bins)*self.t_res, np.sum(self.bins_count[:,:,:,0], axis=(0,1)))
        axs[0].title.set_text("Time [hrs]")
        axs[1].bar(np.arange(self.lon_bins)*self.lon_res, np.sum(self.bins_count[:,:,:,0], axis=(1,2)))
        axs[1].title.set_text("Lon [degrees]")
        axs[2].bar(np.arange(self.lat_bins)*self.lat_res, np.sum(self.bins_count[:,:,:,0], axis=(0,2)))
        axs[2].title.set_text("Lat [degrees]")
        plt.show()

    def plot_variograms(self, variable: AnyStr="u") -> None:
        if self.bins is None:
            raise Exception("Need to run build_variogram() first!")

        variable_map = {"u": 0, "v": 1}

        try:
            var = variable_map[variable]
        except:
            raise ValueError("Specified variable does not exist")

        # find first zero value to avoid dividing by zero error
        t_first_zero_idx = np.argwhere(np.sum(self.bins[:,:,:,var], axis=(0,1)) == 0)[0][0]
        lon_first_zero_idx = np.argwhere(np.sum(self.bins[:,:,:,var], axis=(1,2)) == 0)[0][0]
        lat_first_zero_idx = np.argwhere(np.sum(self.bins[:,:,:,var], axis=(0,2)) == 0)[0][0]

        # plot variograms in one direction
        fig, axs = plt.subplots(1,3,figsize=(25,10))
        axs[0].plot((np.arange(self.t_bins)*self.t_res)[:t_first_zero_idx], np.sum(self.bins[:,:,:,var], axis=(0,1))[:t_first_zero_idx]/(np.sum(self.bins_count[:,:,:,var], axis=(0,1)))[:t_first_zero_idx])
        axs[0].title.set_text("Time [hrs]")
        axs[1].plot((np.arange(self.lon_bins)*self.lon_res)[:lon_first_zero_idx], np.sum(self.bins[:,:,:,var], axis=(1,2))[:lon_first_zero_idx]/(np.sum(self.bins_count[:,:,:,var], axis=(1,2)))[:lon_first_zero_idx])
        axs[1].title.set_text("Lon [degrees]")
        axs[2].plot((np.arange(self.lat_bins)*self.lat_res)[:lat_first_zero_idx], np.sum(self.bins[:,:,:,var], axis=(0,2))[:lat_first_zero_idx]/(np.sum(self.bins_count[:,:,:,var], axis=(0,2)))[:lat_first_zero_idx])
        axs[2].title.set_text("Lat [degrees]")
        plt.show()
