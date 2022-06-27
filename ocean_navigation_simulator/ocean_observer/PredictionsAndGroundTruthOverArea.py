from typing import Union, Dict, Set

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from ocean_navigation_simulator.environment.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.ocean_observer.metrics.observer_metrics import get_metrics
from ocean_navigation_simulator.utils import units


class PredictionsAndGroundTruthOverArea:
    """Class that contains the forecast, improved forecast, uncertainty(if computed) and the hindcast
    for an area
    """

    def __init__(self, predictions_over_area: xr, ground_truth: xr):
        """Create the object using the prediction and ground_truth xarray Datasets
        Specifically, the variables initial_forecast, improved_forecast, and ground_truth_area are calculated as
        3D numpy arrays with T x lat*lon x 2 (water_u and water_v) for metric calculation.

        Args:
            predictions_over_area: xarray Dataset that contains the forecasts, improved forecasts and potentially
            uncertainty
            ground_truth: xarray Dataset that contains the hindcast currents.
        """
        reshape_dims = (len(predictions_over_area["time"]), -1, 2)
        self.predictions_over_area = predictions_over_area
        # interpolate ground_truth to same grid as the predictions
        self.ground_truth = ground_truth.interp_like(predictions_over_area, method='linear')
        self.initial_forecast = np.moveaxis(predictions_over_area[["initial_forecast_u", "initial_forecast_v"]]
                                            .to_array().to_numpy(), 0, -1).reshape(reshape_dims)
        self.improved_forecast = np.moveaxis(predictions_over_area[["water_u", "water_v"]].to_array().to_numpy(), 0,
                                             -1).reshape(reshape_dims)
        self.ground_truth_area = np.moveaxis(self.ground_truth[["water_u", "water_v"]].to_array().to_numpy(), 0,
                                             -1).reshape(reshape_dims)

    def compute_metrics(self, metrics: Union[Set[str], str, None] = None, per_hour: bool = False) -> Dict[str, any]:
        """ Compute and return the metrics provided or all if none is provided.

        Args:
            metrics: The list of metrics names we want to compute.

        Returns:
            A dict containing as key-values metric_id-metric_result
        """
        res = dict()
        for s, f in get_metrics().items():
            if metrics is None or s in metrics:
                res |= f(self.ground_truth_area, self.improved_forecast, self.initial_forecast, per_hour=per_hour)
        return res

    def visualize_improvement_forecasts(self, state_trajectory: np.ndarray, spatial_res=None) -> None:
        """ Display 3 figures representing the initial forecast current map, the improved forecast current map and the
         ground truth current map

        Args:
            state_trajectory: List of points where the platform was
            spatial_res: Spatial resolution of the plots
        """
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('At time {}'.format(
            units.get_datetime_from_np64(self.predictions_over_area['time'][-1])), fontsize=14)
        vmin = 0
        vmax = max(self.initial_forecast.max(), self.improved_forecast.max(), self.ground_truth_area.max())

        ax1 = OceanCurrentSource.plot_data_from_xarray(0, self.ground_truth, vmin=vmin, vmax=vmax, ax=ax[0, 1])
        trajectory_x, trajectory_y = state_trajectory[:, 0], state_trajectory[:, 1]
        x_lim, y_lim = ax1.get_xlim(), ax1.get_ylim()
        ax1.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax1.set_xlim(x_lim), ax1.set_ylim(y_lim)
        ax1.set_title("True currents")

        initial_forecast_reformated = self.predictions_over_area[["initial_forecast_u", "initial_forecast_v"]] \
            .rename(initial_forecast_u="water_u", initial_forecast_v="water_v")
        ax2 = OceanCurrentSource.plot_data_from_xarray(0, initial_forecast_reformated, vmin=vmin, vmax=vmax,
                                                       ax=ax[0, 0])
        x_lim, y_lim = ax2.get_xlim(), ax2.get_ylim()
        ax2.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax2.set_xlim(x_lim), ax2.set_ylim(y_lim)
        ax2.set_title("Initial forecasts")

        ax3 = OceanCurrentSource.plot_data_from_xarray(0, self.predictions_over_area, vmin=vmin, vmax=vmax, ax=ax[1, 1])
        x_lim, y_lim = ax3.get_xlim(), ax3.get_ylim()
        ax3.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax3.set_xlim(x_lim), ax3.set_ylim(y_lim)
        ax3.set_title("Improved forecasts")

        # multiplied by -1 to get error + forecast = hindcast -> Better visually
        error_reformated = self.predictions_over_area[["error_u", "error_v"]] \
                               .rename(error_u="water_u", error_v="water_v") * -1
        magnitude = error_reformated.isel(time=0).assign(magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5)[
            "magnitude"]
        ax4 = OceanCurrentSource.plot_data_from_xarray(0, error_reformated,
                                                       vmin=magnitude.min(),
                                                       vmax=magnitude.max(),
                                                       ax=ax[1, 0])
        x_lim, y_lim = ax4.get_xlim(), ax4.get_ylim()
        ax4.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax4.set_xlim(x_lim), ax4.set_ylim(y_lim)
        ax4.set_title("Error predicted *(-1)")
        fig.tight_layout()

        plt.pause(1)

    def plot_3d(self, variable_to_plot: str) -> None:
        """We plot in 3d the new forecasted values

        Args:
            variable_to_plot: The name of the field from the dataset that we want to plot.
            Could be [error, std_error, initial_forecast, water (for improved forecast), ground_truth]
        """
        label = variable_to_plot
        if variable_to_plot == "ground_truth":
            dataset_to_use = self.ground_truth
            variable_to_plot = "water"
        else:
            dataset_to_use = self.predictions_over_area

        # create list to plot
        data, data_error = [], []
        x_y_intervals = ([min(dataset_to_use["lon"]), max(dataset_to_use["lon"])],
                         [min(dataset_to_use["lat"]), max(dataset_to_use["lat"])])
        for j in range(len(dataset_to_use["time"])):
            elem = dataset_to_use.isel(time=j)
            x, y = np.meshgrid(elem["lon"], elem["lat"])
            z = np.sqrt(elem[variable_to_plot + "_u"].to_numpy() ** 2 + elem[variable_to_plot + "_v"].to_numpy() ** 2)
            times = elem["time"].to_numpy()
            data.append((times, {"X": x, "Y": y, "Z": z}))  # , "colors": "rgy"[j], "alpha": .25})

            # if real_errors is not None:
            #     elem2 = real_errors.isel(time=j)
            #     x, y = np.meshgrid(elem["lon"], elem["lat"])
            #     z = np.sqrt(elem["water_u"].to_numpy() ** 2 + elem["water_v"].to_numpy() ** 2)
            #     times = elem["time"].to_numpy()
            #     data_error.append((times, {"X": x, "Y": y, "Z": z}))  # , "colors": "rgy"[j], "alpha": .25})

        def update_line(idx: int):
            """ Internal function used by the slider to update the 3d wireframe plot

            Args:
                idx: index of the time used for the prediction we plot
            """
            ax.clear()
            t, dic = data[idx]
            ax.plot_wireframe(**dic)
            plt.draw()
            if len(data_error):
                t2, dic2 = data_error[idx]
                ax.plot_wireframe(**dic2, alpha=0.5, color='g')
            ax.set_title(
                f"{label}\nCurrent time:{np.datetime_as_string(data[0][0], unit='s')}\n" +
                f"Prediction time:{np.datetime_as_string(t, unit='s')}\n" +
                f"Mean {dic['Z'].mean()}, std: {dic['Z'].std()}")
            if x_y_intervals:
                ax.set_xlim(xmin=x_y_intervals[0][0], xmax=x_y_intervals[0][1])
                ax.set_ylim(ymin=x_y_intervals[1][0], ymax=x_y_intervals[1][1])
            plt.draw()
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            ax.set_zlabel(label)

            # Plot the trajectory of the boat:
            # Neet do recompute the magnitude using the hindcast at the platform past and present positions
            # magnitude_error = ...
            # ax.plot(state_trajectory[::stride, 0], state_trajectory[::stride, 1],
            #         magnitude_error, c='black')
            # ax.scatter(state_trajectory[::stride, 0], state_trajectory[::stride, 1],
            #            magnitude_error,
            #            marker=".")

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        update_line(0)
        # Plot the Slider
        plt.subplots_adjust(bottom=.25)
        ax_slider = plt.axes([.1, .1, .8, .05], facecolor='teal')
        slider = Slider(ax_slider, "Time", valmin=0, valmax=len(data) - 1, valinit=0, valstep=1)
        slider.on_changed(update_line)

        plt.show()
        keyboard_click = False
        while not keyboard_click:
            print("waiting for keyboard input to continue")
            keyboard_click = plt.waitforbuttonpress()
            print("continue scenario")
