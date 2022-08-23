from datetime import datetime, timezone
from typing import Union, Dict, Set, Optional, List, Tuple

import numpy as np
import xarray as xr
from DateTime import DateTime
from matplotlib import pyplot as plt, patches
from matplotlib.widgets import Slider, Button

from ocean_navigation_simulator.data_sources import OceanCurrentSource
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

    def compute_metrics(self, metrics: Union[Set[str], str, None] = None, directions: list[str] = ['uv'],
                        per_hour: bool = False) -> Dict[str, any]:
        """ Compute and return the metrics provided or all if none is provided.

        Args:
            metrics: The list of metrics names we want to compute.

        Returns:
            A dict containing as key-values metric_id-metric_result
        """
        res = dict()
        for s, f in get_metrics().items():
            for d in directions:
                if not check_nans(self.ground_truth_area, self.improved_forecast, current=d):
                    if metrics is None or s in set(metrics):
                        res |= f(self.ground_truth_area, self.improved_forecast, self.initial_forecast,
                                 per_hour=per_hour,
                                 current=d)
        return res

    def visualize_initial_error(self, list_predictions: List[Tuple['xr', 'xr']], spatial_res=None,
                                tuple_trajectory_history_new_files: Optional[Tuple[np.array, List[DateTime]]] = None,
                                radius_area: float = None, gp_outputs: Optional[List['xr']] = None):
        print("visualize initial error")
        # init_fc = self.predictions_over_area[["initial_forecast_u", "initial_forecast_v"]].rename(
        #     {"initial_forecast_u": "water_u", "initial_forecast_v": "water_v"})
        # error = init_fc - self.ground_truth
        if gp_outputs is None:
            errors_predicted = [pred[0][["error_u", "error_v"]].rename(
                {"error_u": "water_u", "error_v": "water_v"}) for pred in list_predictions]
        else:
            errors_predicted = [pred[["error_u", "error_v"]].rename(
                {"error_u": "water_u", "error_v": "water_v"}) for pred in gp_outputs]
            std_output = [pred[["std_error_u", "std_error_v"]].rename(
                {"std_error_u": "water_u", "std_error_v": "water_v"}) for pred in gp_outputs]

        initial_forecasts = [pred[0][["initial_forecast_u", "initial_forecast_v"]].rename(
            {"initial_forecast_u": "water_u", "initial_forecast_v": "water_v"}).assign(
            magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5) for pred in
            list_predictions]
        real_errors = [(pred[0][["initial_forecast_u", "initial_forecast_v"]].rename(
            {"initial_forecast_u": "water_u", "initial_forecast_v": "water_v"}) - pred[1]).assign(
            magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5) for pred in
            list_predictions]

        fig, ax = plt.subplots(2, 2)
        ax1, ax2, ax3, ax4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

        ax1, self.cbar = OceanCurrentSource.plot_data_from_xarray(0, real_errors[0], ax=ax1,
                                                                  return_cbar=True)
        ax2 = OceanCurrentSource.plot_data_from_xarray(0, initial_forecasts[0], ax=ax2, colorbar=False)
        ax3, self.cbar_3 = OceanCurrentSource.plot_data_from_xarray(0, errors_predicted[0], ax=ax3, return_cbar=True)
        ax4, self.cbar_4 = OceanCurrentSource.plot_data_from_xarray(0, (std_output[0] if gp_outputs is not None else
                                                                        errors_predicted[0]), ax=ax4, return_cbar=True)

        # x_lim, y_lim = ax1.get_xlim(), ax1.get_ylim()

        # Slider

        def update_maps(lag, index_prediction, ax1, ax2, ax3, ax4):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            self.cbar.remove()
            self.cbar_3.remove()
            fig.suptitle('At time {}, with a lag of {} hours.'.format(
                units.get_datetime_from_np64(initial_forecasts[index_prediction]['time'][0]), lag), fontsize=14)
            vmin, vmax = 0, float(max(initial_forecasts[index_prediction].isel(time=lag)["magnitude"].max(),
                                      real_errors[index_prediction].isel(time=lag)["magnitude"].max()))
            _, self.cbar = OceanCurrentSource.plot_data_from_xarray(lag, real_errors[index_prediction], ax=ax1,
                                                                    return_cbar=True, vmin=vmin, vmax=vmax)
            OceanCurrentSource.plot_data_from_xarray(lag, initial_forecasts[index_prediction], ax=ax2, colorbar=False,
                                                     vmin=vmin, vmax=vmax)
            _, self.cbar_3 = OceanCurrentSource.plot_data_from_xarray(lag, errors_predicted[index_prediction], ax=ax3,
                                                                      return_cbar=True)
            ax1.set_title("Error forecast vs hindcast [m/s]")
            ax2.set_title("Forecast currents [m/s]")
            ax3.set_title("Error currents predicted by the GP [m/s]")
            ax4.set_title("std predicted by the GP (zoomed in) [m/s]")

            # Print trajectory
            if tuple_trajectory_history_new_files is not None:
                trajectory, new_files = tuple_trajectory_history_new_files
                datetime_day_selected = units.get_datetime_from_np64(
                    initial_forecasts[index_prediction].isel(time=0)["time"])

                # # index_latest_position_trajectory = max(0, np.argmax(time_prediction_ts < trajectory[:, 2]) - 1)
                # index_latest_position_trajectory_before = np.sum(timestamp_day_selected > trajectory[:, 2]) - 1
                # date_latest_position_trajectory_before = datetime.fromtimestamp(
                #     trajectory[index_latest_position_trajectory_before][2], tz=timezone.utc)

                last_date_before = None
                latest_position = trajectory[0]
                for pos in trajectory:
                    date_trajectory = datetime.fromtimestamp(pos[2], tz=timezone.utc)
                    if date_trajectory < datetime_day_selected:
                        last_date_before = date_trajectory
                        latest_position = pos
                    else:
                        break

                last_file_date_before = None
                for file_date_file in new_files:
                    if file_date_file < datetime_day_selected:
                        last_file_date_before = file_date_file
                    else:
                        break
                if radius_area is not None:
                    x, y = latest_position[:2]
                    error_small_window = errors_predicted[index_prediction].sel(
                        lon=slice(x - radius_area / 2, x + radius_area),
                        lat=slice(y - radius_area / 2, y + radius_area))
                    # print("error:", error_small_window)
                    self.cbar_4.remove()
                    _, self.cbar_4 = OceanCurrentSource.plot_data_from_xarray(lag, std_output[
                        index_prediction] if gp_outputs is not None else error_small_window, ax=ax4, return_cbar=True)
                    for ax in [ax1, ax2, ax3, ax4]:
                        rect = patches.Rectangle((x - radius_area, y - radius_area), radius_area * 2,
                                                 radius_area * 2,
                                                 linewidth=1,
                                                 edgecolor='r',
                                                 facecolor='none')
                        ax.add_patch(rect)
                for ax in [ax1, ax2, ax3, ax4]:
                    for pt in trajectory:
                        date_timestamp = pt[2]
                        date_pt = datetime.fromtimestamp(date_timestamp, tz=timezone.utc)
                        if date_pt > datetime_day_selected:
                            break

                        if last_file_date_before is not None and last_file_date_before > date_pt:
                            color = 'red'
                            ax.scatter(pt[0], pt[1], c=color)
                        elif last_date_before is not None and last_date_before > date_pt:
                            color = 'yellow'
                            ax.scatter(pt[0], pt[1], c=color)

        time_dim = list(range(len(list_predictions[-1][0]["time"])))

        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Make a horizontal slider to control the frequency.
        ax_lag_time = plt.axes([0.25, 0.1, 0.65, 0.03])
        lag_time_slider = Slider(
            ax=ax_lag_time,
            label='Lag (in hours) with forecast',
            valmin=min(time_dim),
            valmax=max(time_dim) - 1,
            valinit=time_dim[0],
            valstep=1
        )

        # Make a vertically oriented slider to control the amplitude
        axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
        forecast_slider = Slider(
            ax=axamp,
            label="Day forecast",
            valmin=0,
            valmax=len(list_predictions) - 1,
            valinit=0,
            orientation="vertical",
            valstep=1
        )

        # The function to be called anytime a slider's value changes
        def update(_):
            update_maps(lag_time_slider.val, forecast_slider.val, ax1, ax2, ax3, ax4)
            fig.canvas.draw_idle()

        # register the update function with each slider
        lag_time_slider.on_changed(update)
        forecast_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            ax_lag_time.reset()
            forecast_slider.reset()

        button.on_clicked(reset)
        update_maps(0, 0, ax1, ax2, ax3, ax4)
        plt.show()
        keyboardClick = False
        while keyboardClick != True:
            keyboardClick = plt.waitforbuttonpress()

    def visualize_improvement_forecasts(self, state_trajectory: Optional[np.ndarray] = None, spatial_res=None) -> None:
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
        vmax = max(np.nanmax(self.initial_forecast), np.nanmax(self.improved_forecast),
                   np.nanmax(self.ground_truth_area))

        ax1 = OceanCurrentSource.plot_data_from_xarray(0, self.ground_truth, vmin=vmin, vmax=vmax, ax=ax[0, 1])
        x_lim, y_lim = ax1.get_xlim(), ax1.get_ylim()
        ax1.set_xlim(x_lim), ax1.set_ylim(y_lim)
        ax1.set_title("True currents")
        if state_trajectory is not None:
            trajectory_x, trajectory_y = state_trajectory[:, 0], state_trajectory[:, 1]
            ax1.plot(trajectory_x, trajectory_y, color='y', marker='+')

        initial_forecast_reformated = self.predictions_over_area[["initial_forecast_u", "initial_forecast_v"]] \
            .rename(initial_forecast_u="water_u", initial_forecast_v="water_v")
        ax2 = OceanCurrentSource.plot_data_from_xarray(0, initial_forecast_reformated, vmin=vmin, vmax=vmax,
                                                       ax=ax[0, 0])
        x_lim, y_lim = ax2.get_xlim(), ax2.get_ylim()
        if state_trajectory is not None:
            ax2.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax2.set_xlim(x_lim), ax2.set_ylim(y_lim)
        ax2.set_title("Initial forecasts")

        ax3 = OceanCurrentSource.plot_data_from_xarray(0, self.predictions_over_area, vmin=vmin, vmax=vmax, ax=ax[1, 1])
        x_lim, y_lim = ax3.get_xlim(), ax3.get_ylim()
        if state_trajectory is not None:
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
        if state_trajectory is not None:
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
