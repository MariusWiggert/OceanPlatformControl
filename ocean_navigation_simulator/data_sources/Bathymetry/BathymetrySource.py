import logging
import os
import time
from typing import AnyStr, Dict, List, Optional

import casadi as ca
import cmocean
import matplotlib.pyplot
import matplotlib.pyplot as plt
import xarray as xr

# from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.data_sources.DataSource2d import DataSource2d
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint


class BathymetrySource2d(DataSource2d):
    def __init__(self, source_dict: Dict):
        self.elevation_func = None  # Casadi function
        super().__init__(source_dict)
        self.logger = logging.getLogger("areana.bathymetry_source_2d")
        self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

    def instantiate_source_from_dict(self) -> None:
        if self.source_dict["source"] == "gebco":
            self.DataArray = self.get_DataArray_from_file()
            self.grid_dict = self.get_grid_dict_from_xr(self.DataArray)
        else:
            raise NotImplementedError(
                f"Selected source {self.source_dict['source']} in the BathymetrySource dict is not implemented."
            )

    def get_DataArray_from_file(self) -> xr:
        DataArray = xr.open_dataset(self.source_dict["source_settings"]["filepath"])
        # DataArray = DataArray.rename({"latitude": "lat", "longitude": "lon"})
        return DataArray

    def get_data_at_point(self, spatial_point: SpatialPoint) -> float:
        # Invert spatial point order to (lat, lon)
        return self.elevation_func(spatial_point.__array__()[::-1])

    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        # Note: the input to the casadi function needs to be an array of the form np.array([lat, lon])
        Args:
          grid:     list of the 2 grids [y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """

        self.elevation_func = ca.interpolant(
            "elevation", "linear", grid, array["elevation"].values.ravel(order="F")
        )

    def is_higher_than(self, point: SpatialPoint, elevation: float = 0):
        """Helper function to check if a SpatialPoint is on the land.
            Accuracy is limited by the resolution of self.grid_dict.
        Args:
            point:    SpatialPoint object where to check if it is on land
        Returns:
            bool:     True if on land and false otherwise
        """

        if not (
            self.casadi_grid_dict["x_range"][0]
            < point.lon.deg
            < self.casadi_grid_dict["x_range"][1]
        ):
            raise ValueError(
                f"Point {point} is not in casadi_grid_dict lon range{self.casadi_grid_dict['x_range']}"
            )

        if not (
            self.casadi_grid_dict["y_range"][0]
            < point.lat.deg
            < self.casadi_grid_dict["y_range"][1]
        ):
            raise ValueError(
                f"Point {point} is not in casadi_grid_dict lat range {self.casadi_grid_dict['y_range']}"
            )

        return self.get_data_at_point(point) > elevation

    @staticmethod
    def plot_data_from_xarray(
        xarray: xr,
        var_to_plot: AnyStr = "elevation",
        vmin: Optional[float] = -6000,
        vmax: Optional[float] = 6000,
        alpha: Optional[float] = 1.0,
        ax: plt.axes = None,
        fill_nan: bool = True,
    ) -> matplotlib.pyplot.axes:
        """Bathymetry specific plotting function to plot the x_array.
        All other functions build on top of it, it creates the ax object and returns it.
        Args:
            time_idx:          time-idx to select from the xarray (only if it has time dimension)
            xarray:            xarray object containing the grids and data
            var_to_plot:       a string of the variable to plot
            vmin:              minimum current magnitude used for colorbar (float)
            vmax:              maximum current magnitude used for colorbar (float)
            alpha:             alpha of the current magnitude color visualization
            ax:                Optional for feeding in an axis object to plot the figure on.
            fill_nan:          Optional if True we fill nan values with 0 otherwise leave them as nans.
        Returns:
            ax                 matplotlib.pyplot.axes object
        """
        if fill_nan:
            xarray = xarray.fillna(0)
        if ax is None:
            ax = plt.axes()

        # plot data for the specific variable
        # if vmax is None:
        #     vmax = xarray[var_to_plot].max()
        # if vmin is None:
        #     vmin = xarray[var_to_plot].min()
        # TODO: think of smart structure
        # Fix colorbar limits, as land will be covered by platecarree land map
        # if we use geographic coordinate system and we don't need it for land
        cmap = cmocean.cm.topo
        xarray[var_to_plot].plot(cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, ax=ax)
        # Label the plot
        ax.set_title(
            "Variable: {var} \n at Time: {t}".format(
                var=var_to_plot, t="Time: " + time.strftime("%Y-%m-%d %H:%M:%S UTC")
            )
        )
        return ax

    def __del__(self):
        """Helper function to delete the existing casadi functions."""
        del self.elevation_func
        pass
