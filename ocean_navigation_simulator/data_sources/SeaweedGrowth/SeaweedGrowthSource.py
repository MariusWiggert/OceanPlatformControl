import datetime
import logging
from typing import Dict, List

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ocean_navigation_simulator.data_sources.DataSource import (
    AnalyticalSource,
    DataSource,
)
from ocean_navigation_simulator.data_sources.SeaweedGrowth.SeaweedFunction import (
    compute_R_growth_without_irradiance,
    compute_R_resp,
    irradianceFactor,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.utils import units

# TODO: Automatically handle re-initialization of the F_NGR_per_second casadi function when the solar_rad_casadi
# in the solar_source is updated (e.g. because of caching). Needs to happen either in Arena or Platform.
# => this works now without issues when we use a non-caching SolarIrradianceSource e.g. AnalyticalSolarIrradiance
# From Experiments: Interpolation extrapolates outside of it's domain using the dy/dx at the boundary of the domain.
# When the interpolation function is updated, the outside function is NOT updated! So we need to re-run set_casadi_function
# every time the solar function does new caching!


# Not used right now as different modality as other sources.
class SeaweedGrowthSource(DataSource):
    """Base class for the Seaweed Growth data sources.
    Note: It requires input from the Solar Source in W/m^2 and
    """

    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """
        raise NotImplementedError


class SeaweedGrowthGEOMAR(SeaweedGrowthSource, AnalyticalSource):
    """Seaweed Growth based on the model from the paper below with various simplifications for our use-case (see Notion).
    Set 'source' = 'GEOMAR_paper' for instantiating this source in the SeaweedGrowthField
    Wu, Jiajun, David P. Keller, and Andreas Oschlies.
    "Carbon Dioxide Removal via Macroalgae Open-ocean Mariculture and Sinking: An Earth System Modeling Study."
     Earth System Dynamics Discussions (2022): 1-52.

     The nutrient concentrations of phosphate and nitrate as well as temperatures are taken
     from 2021 monthly averaged nutrient data from Copernicus."""

    def __init__(self, source_config_dict: Dict):
        """Dictionary with the three top level keys:
         'field' the kind of field the should be created, here SeaweedGrowth
         'source' = 'GEOMAR' for instantiating this source
         'source_settings':{
            'filepath': './data/2021_monthly_nutrients_and_temp.nc'
            'solar_func_ca': pointer to the casadi function from the solar module to use here.
        }
        """
        # Adding things that are required for the AnalyticalSource to the dictionary
        source_config_dict = self.add_default_domains(source_config_dict)
        super().__init__(source_config_dict)
        # initialize logger
        self.logger = logging.getLogger("arena.seaweed_field.seaweed_growth_source")

        # Initialize variables used to hold casadi functions.
        self.F_NGR_per_second, self.r_growth_wo_irradiance, self.r_resp = [None] * 3
        self.solar_rad_casadi = source_config_dict["source_settings"][
            "solar_source"
        ].solar_rad_casadi

        # Open the nutrient dataset and calculate the derived values from it
        self.DataArray = self.get_growth_and_resp_data_array_from_file()
        # Set-up the interpolation functions (valid for all years and space)
        self.set_casadi_function()
        # Initialize the super function
        self.grid_dict = self.get_grid_dict_from_xr(self.DataArray)
        self.casadi_grid_dict = self.grid_dict

    def get_growth_and_resp_data_array_from_file(self) -> xr:
        """Helper function to open the dataset and calculate the metrics derived from nutrients and temp."""
        # TODO: Clean up with parameters like year etc! we can actually just pre-compute it and just load the nc file for those two values.
        path = "./data/seaweed/seaweed_precomputed.nc"
        # if os.path.exists(path):
        #     return xr.open_dataset(path)
        # else:
        DataArray = xr.open_dataset(self.source_config_dict["source_settings"]["filepath"])
        DataArray = DataArray.rename({"latitude": "lat", "longitude": "lon"})
        DataArray = DataArray.assign(
            R_growth_wo_Irradiance=compute_R_growth_without_irradiance(
                DataArray["Temperature"], DataArray["no3"], DataArray["po4"]
            )
        )
        DataArray = DataArray.assign(R_resp=compute_R_resp(DataArray["Temperature"]))
        # Just to conserve RAM
        DataArray = DataArray.drop(["Temperature", "no3", "po4"])
        # DataArray.to_netcdf(path=path)

        return DataArray

    @staticmethod
    def posix_to_rel_sec_in_year_ca(posix_timestamp: float) -> float:
        """Helper function to map a posix_timestamp to it's relative seconds for the specific year (since 1st of January).
        This is needed because the interpolation function for the nutrients operates on relative timestamps as we take
        the average monthly nutrients for those as input.
        Args:
            posix_timestamp: a posix timestamp
        """
        correction_seconds = 13 * 24 * 3600
        # Calculate the relative time of the year in seconds
        return ca.mod(posix_timestamp - correction_seconds, 365 * 24 * 3600)

    def set_casadi_function(self):
        """Creates the symbolic computation graph for the full casadi function."""
        # Step 1: Create the grid for interpolation (in relative seconds per year, not posix time as other sources)
        grid = [
            units.posix_to_rel_seconds_in_year(
                units.get_posix_time_from_np64(self.DataArray.coords["time"].values)
            ),
            self.DataArray.coords["lat"].values,
            self.DataArray.coords["lon"].values,
        ]
        # Set-up the casadi interpolation functions for the growth and respiration factors
        r_growth_wo_irradiance = ca.interpolant(
            "r_growth_wo_irradiance",
            "linear",
            grid,
            self.DataArray["R_growth_wo_Irradiance"].fillna(0).values.ravel(order="F"),
        )
        r_resp = ca.interpolant(
            "r_resp", "linear", grid, self.DataArray["R_resp"].fillna(0).values.ravel(order="F")
        )

        # Step 2: Set-Up the casadi graph for the full calculation
        sym_time = ca.MX.sym("time")  # in posix
        sym_lon_degree = ca.MX.sym("lon")  # in deg
        sym_lat_degree = ca.MX.sym("lat")  # in deg

        # Calculate solar factor
        sym_irradiance = self.solar_rad_casadi(ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree))
        sym_f_Irradiance = irradianceFactor(sym_irradiance)
        # put things together to the NGR
        sym_rel_year_time = self.posix_to_rel_sec_in_year_ca(sym_time)
        sym_rel_year_input = ca.vertcat(sym_rel_year_time, sym_lat_degree, sym_lon_degree)
        sym_NGR_per_second = r_growth_wo_irradiance(sym_rel_year_input) / (
            24 * 3600
        ) * sym_f_Irradiance - r_resp(sym_rel_year_input) / (24 * 3600)
        # Set up the full function
        self.F_NGR_per_second = ca.Function(
            "d_biomass_dt_in_seconds",
            [ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree)],
            [sym_NGR_per_second],
        )

    def plot_R_growth_wo_Irradiance(self, time: datetime.datetime):
        """Helper Function to visualize the R_growth_wo_Irradiance per day at a time."""
        Map_to_plot = self.DataArray["R_growth_wo_Irradiance"].interp(
            time=np.datetime64(time.replace(tzinfo=None))
        )
        Map_to_plot.plot()
        plt.title("Seaweed Growth Factor without Irradiance adjusting per day.")
        plt.show()

    def create_xarray(self, grids_dict: Dict, NGR_per_second: np.array) -> xr:
        """Function to create an xarray from the data tuple and grid dict
        Args:
          NGR_per_second: tuple containing the data of the source as numpy array [T, Y, X]
          grids_dict: containing ranges and grids of x, y, t dimension
        Returns:
          xr     an xarray containing both the grid and data
        """
        array = xr.Dataset(
            dict(F_NGR_per_second=(["time", "lat", "lon"], NGR_per_second)),
            coords=dict(
                lon=grids_dict["x_grid"],
                lat=grids_dict["y_grid"],
                time=np.round(np.array(grids_dict["t_grid"]) * 1000, 0).astype("datetime64[ms]"),
            ),
        )

        array["F_NGR_per_second"].attrs = {
            "units": "Net Growth Rate Factor per second (*Biomass to get the dMass/dt) "
        }
        return array

    def map_analytical_function_over_area(self, grids_dict: Dict) -> np.array:
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
        Args:
          grids_dict: containing grids of x, y, t dimension
        Returns:
          data     containing the data in whatever format as numpy array (not yet in xarray form) e.g. Tuple
        """
        # Step 1: Create the meshgrid numpy matrices for each coordinate
        LAT, TIMES, LON = np.meshgrid(
            grids_dict["y_grid"], grids_dict["t_grid"], grids_dict["x_grid"]
        )
        #LAT, LON = np.meshgrid(grids_dict["y_grid"], grids_dict["x_grid"])

        data_out = self.F_NGR_per_second(ca.DM([TIMES.flatten(), LAT.flatten(), LON.flatten()]))
        # print("data",data_out)
        # print("reshaped",np.array(data_out).reshape(LAT.shape))
        # return reshaped to proper size

        # LON, LAT = np.where((LON >= -82.2) & (LON <= -81.6), 1, 0), np.where(
        #     (LAT >= 23.7) & (LAT <= 24.3), 1, 0
        # )
        # data = np.multiply(LON.T, LAT.T)

        # T = grids_dict["t_grid"].shape[0]

        # data = np.repeat(data[np.newaxis, :, :], T, axis=0)
        return np.array(data_out).reshape(LAT.shape)

        return data

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> float:
        """Function to get the data at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          float of the net_growth_rate per second
        """
        return self.F_NGR_per_second(SpatioTemporalPoint.to_spatio_temporal_casadi_input())

    @staticmethod
    def add_default_domains(source_config_dict: Dict) -> Dict:
        """Helper Function to make it work smoothly with the AnalyticalSource class."""
        source_config_dict["source_settings"]["x_domain"] = [-180, 180]
        source_config_dict["source_settings"]["y_domain"] = [-90, 90]
        source_config_dict["source_settings"]["temporal_domain"] = [
            datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2024, 1, 10, 0, 0, 0, tzinfo=datetime.timezone.utc),
        ]
        source_config_dict["source_settings"]["spatial_resolution"] = 0.1
        source_config_dict["source_settings"]["temporal_resolution"] = 3600

        return source_config_dict

    def update_casadi_dynamics(self, state: PlatformState) -> None:
        pass
