from Variogram import Variogram
from ocean_navigation_simulator.generative_error_model.utils import load_config


class VisualizeVariogram:

    def __init__(self, variogram:Variogram=None):
        self.variogram = variogram


    def read_variogram_from_file(self, file_name: str=None):
        self.config = load_config()["variogram"]
        if file_name is None:
            file_path = os.path.join(self.config["data_dir"], self.config["file_name"])
        else:
            file_path = os.path.join(self.config["data_dir"], file_name)
        pass


    def plot_histograms(self, tol: Tuple[int]=(10,10,5)) -> None:
        """Plots the histogram of bins in each axis [lon, lat, time]."""

        if self.bins is None:
            raise Exception("Need to run build_variogram() first!")

        # plot histogram for each axis
        fig, axs = plt.subplots(1,3,figsize=(25,10))
        axs[0].bar(np.arange(self.lon_bins)*self.lon_res, np.sum(self.bins_count[:,:tol[1],:tol[2],0], axis=(1,2)))
        axs[0].set_xlabel("Lon [degrees]")
        axs[0].set_ylabel("Frequency")
        axs[1].bar(np.arange(self.lat_bins)*self.lat_res, np.sum(self.bins_count[:tol[0],:,:tol[2],0], axis=(0,2)))
        axs[1].set_xlabel("Lat [degrees]")
        axs[2].bar(np.arange(self.t_bins)*self.t_res, np.sum(self.bins_count[:tol[0],:tol[1],:,0], axis=(0,1)))
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


    def plot_variograms(self, variable: AnyStr="u", tol: Tuple[int]=(10,10,5)) -> None:
        """Plots the sliced variogram for each axis [lon, lat, time]."""

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
        lon_y_num = np.sum(self.bins[:,:tol[1],:tol[2],var], axis=(1,2))
        # lon_y_denom = self.bins.shape[1]*self.bins.shape[2]
        lon_y_denom = tol[1]*tol[2]
        axs[0].scatter(np.arange(self.lon_bins)*self.lon_res, np.divide(lon_y_num, lon_y_denom, \
            out=np.zeros_like(lon_y_num), where=lon_y_denom!=0), marker="x")
        axs[0].set_xlabel("Lon lag [degrees]")
        axs[0].set_ylabel("Semivariance")

        lat_y_num = np.sum(self.bins[:tol[0],:,:tol[2],var], axis=(0,2))
        # lat_y_denom = self.bins.shape[0]*self.bins.shape[2]
        lat_y_denom = tol[0]*tol[2]
        axs[1].scatter(np.arange(self.lat_bins)*self.lat_res, np.divide(lat_y_num, lat_y_denom, \
            out=np.zeros_like(lat_y_num), where=lat_y_denom!=0), marker="x")
        axs[1].set_xlabel("Lat lag [degrees]")

        t_y_num = np.sum(self.bins[:tol[0],:tol[1],:,var], axis=(0,1))
        # t_y_denom = self.bins.shape[0]*self.bins.shape[1]
        t_y_denom = tol[0]*tol[1]
        axs[2].scatter(np.arange(self.t_bins)*self.t_res, np.divide(t_y_num, t_y_denom, \
            out=np.zeros_like(t_y_num), where=t_y_denom!=0), marker="x")
        axs[2].set_xlabel("Time lag [hrs]")
        plt.show() 
   