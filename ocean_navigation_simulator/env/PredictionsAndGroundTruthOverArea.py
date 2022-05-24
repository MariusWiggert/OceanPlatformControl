import datetime
from typing import Union, Dict, Set, Optional, AnyStr, Tuple

import matplotlib.axes
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.utils.metrics import get_metrics


class PredictionsAndGroundTruthOverArea:

    def __init__(self, predictions_over_area: xr, ground_truth: xr):
        reshape_dims = (len(predictions_over_area["time"]), -1, 2)
        self.predictions_over_area = predictions_over_area
        self.ground_truth = ground_truth
        self.initial_forecast = np.moveaxis(predictions_over_area[["initial_forecast_u", "initial_forecast_v"]]
                                            .to_array().to_numpy(), 0, -1).reshape(reshape_dims)
        self.improved_forecast = np.moveaxis(predictions_over_area[["water_u", "water_v"]].to_array().to_numpy(), 0,
                                             -1).reshape(reshape_dims)
        self.ground_truth_area = np.moveaxis(ground_truth[["water_u", "water_v"]].to_array().to_numpy(), 0, -1).reshape(
            reshape_dims)

    def get_improved_error(self) -> xr:
        return xr.merge([])

    def compute_metrics(self, metrics: Union[Set[str], str, None] = None) -> Dict[str, any]:
        res = dict()
        for s, f in get_metrics().items():
            if metrics is None or s in metrics:
                res |= f(self.ground_truth_area, self.improved_forecast, self.initial_forecast)
        return res

    def _plot_xarray(self, dataset_to_plot: xr, time: Union['datetime.datetime', float] = None,
                     return_ax: Optional[bool] = False, **kwargs) -> Union[matplotlib.axes.Axes, None]:
        """Plot the data at a specific time over an area defined by the x and y intervals.
        Args:
          time: time for which to plot the data either posix or datetime.datetime object
          x_interval:       List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
          y_interval:       List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
          spatial_res:      Per default (None) the data_source resolution is used otherwise the selected one.
          error_to_incorporate: The error to add to the forecast. It assumes to have the correct shape and resolution if
                                it is a xr.DataArray or it can simply be a datasource.
          return_ax:         if True returns ax, otherwise renders plots with plt.show()
          **kwargs:          Further keyword arguments for more specific setting, see plot_currents_from_2d_xarray.
        """

        # format to datetime object
        if not isinstance(time, datetime.datetime):
            time = datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc)

        # Step 1: get the area data

        # interpolate to specific time
        at_time_array = dataset_to_plot.interp(time=time.replace(tzinfo=None))
        # Plot the current field
        ax = self.plot_data_from_xarray(time_idx=0, xarray=at_time_array, **kwargs)

        if return_ax:
            return ax
        else:
            plt.show()

    # Todo: to remove!!!
    def plot_data_from_xarray(self, time_idx: int, xarray: xr, var_to_plot: AnyStr = None,
                              vmin: Optional[float] = None, vmax: Optional[float] = None,
                              alpha: Optional[float] = 1., reset_plot: Optional[bool] = False,
                              figsize: Tuple[int] = (6, 6)) -> matplotlib.pyplot.axes:
        """Base function to plot a specific var_to_plot of the x_array. If xarray has a time-dimension time_idx is selected,
        if xarray's time dimension is already collapsed (e.g. after interpolation) it's directly plotted.
        All other functions build on top of it, it creates the ax object and returns it.
        Args:
            time_idx:          time-idx to select from the xarray (only if it has time dimension)
            xarray:            xarray object containing the grids and data
            var_to_plot:       a string of the variable to plot
            vmin:              minimum current magnitude used for colorbar (float)
            vmax:              maximum current magnitude used for colorbar (float)
            alpha:             alpha of the current magnitude color visualization
            reset_plot:        if True the current figure is re-setted otherwise a new figure created (used for animation)
            figsize:           size of the figure
        Returns:
            ax                 matplotlib.pyplot.axes object
        """
        # reset plot this is needed for matplotlib.animation
        if reset_plot:
            plt.clf()
        else:  # create a new figure object where this is plotted
            fig = plt.figure(figsize=figsize)

        # Get data variable if not provided
        if var_to_plot is None:
            var_to_plot = list(xarray.keys())[0]

        # Step 1: Make the data ready for plotting
        # check if time-dimension already collapsed or not yet
        if xarray['time'].size != 1:
            xarray = xarray.isel(time=time_idx)
        time = units.get_datetime_from_np64(xarray['time'].data)

        # Step 2: Create ax object
        # Non-dimensional
        ax = plt.axes()
        time_string = "Time: {time:.2f}".format(time=time.timestamp())

        # plot data for the specific variable
        if vmax is None:
            vmax = xarray[var_to_plot].max()
        if vmin is None:
            vmin = xarray[var_to_plot].min()
        xarray[var_to_plot].plot(cmap='viridis', vmin=vmin, vmax=vmax, alpha=alpha, ax=ax)

        # Label the title
        ax.set_title("Field: {f} \n Variable: {var} \n at Time: {t}".format(
            f=self.source_config_dict['field'],
            var=var_to_plot,
            t=time_string))

        return ax

    def visualize_improved_error(self, state_trajectory: np.ndarray, spatial_res=None) -> None:
        print("visualize_currents")
        return
        # todo: make it adaptable by taking min max over the 3 sources
        vmin, vmax = -1, 1
        t = pd.to_datetime(self.ground_truth["time"][0].to_numpy())
        ax1 = self.arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
            self.last_observation.platform_state.date_time, *x_y_intervals, return_ax=True, vmin=vmin, vmax=vmax)
        ax1 = self._plot_xarray(self.ground_truth, time=t, return_ax=True)
        trajectory_x, trajectory_y = state_trajectory[:, 0], state_trajectory[:, 1]
        x_lim, y_lim = ax1.get_xlim(), ax1.get_ylim()
        ax1.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax1.set_xlim(x_lim), ax1.set_ylim(y_lim)
        ax1.set_title("True currents")

        ax2 = self.arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
            self.last_observation.platform_state.date_time,
            *x_y_intervals,
            return_ax=True, vmin=vmin, vmax=vmax)
        x_lim, y_lim = ax2.get_xlim(), ax2.get_ylim()
        ax2.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax2.set_xlim(x_lim), ax2.set_ylim(y_lim)
        ax2.set_title("Initial forecasts")

        error_reformated = self.last_prediction[["mean_error_u", "mean_error_v"]].rename(mean_error_u="water_u",
                                                                                         mean_error_v="water_v")
        ax3 = self.arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
            self.last_observation.platform_state.date_time,
            *x_y_intervals,
            error_to_incorporate=error_reformated,
            return_ax=True, vmin=vmin, vmax=vmax)
        x_lim, y_lim = ax3.get_xlim(), ax3.get_ylim()
        ax3.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax3.set_xlim(x_lim), ax3.set_ylim(y_lim)
        ax3.set_title("Improved forecasts")

        vmin = min(ax1.g)
