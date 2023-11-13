import abc
import datetime
import glob
import logging
from typing import Dict, List, Union
import os

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

    def create_xarray(self, grids_dict: Dict, data_tuple: np.array) -> xr:
        """Function to create an xarray from the data tuple and grid dict
        Args:
          data_tuple/NGR_per_second: np.array containing the data of the data [T, Y, X]
          grids_dict: containing ranges and grids of x, y, t dimension
        Returns:
          xr     an xarray containing both the grid and data
        """
        array = xr.Dataset(
            dict(F_NGR_per_second=(["time", "lat", "lon"], data_tuple)),
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


class SeaweedGrowthAnalytical(SeaweedGrowthSource, AnalyticalSource):
    """Analytical Seaweed Growth Source base class.
    Children classes need to only implement the F_NGR_per_second_analytical function.
    """

    def __init__(self, source_config_dict: Dict):
        super().__init__(source_config_dict)
        # initialize logger
        self.logger = logging.getLogger("arena.seaweed_field.seaweed_growth_source")

    @abc.abstractmethod
    def F_NGR_per_second_analytical(
            self,
            lon: Union[float, np.array],
            lat: Union[float, np.array],
            posix_time: Union[float, np.array],
    ) -> Union[float, np.array]:
        """Calculating the current in longitudinal direction at a specific lat, lon point at posix_time.
        Note: this can be used for floats as input or np.arrays which are all the same shape.
        Args:
            lon: longitude in degree
            lat: latitude in degree
            posix_time: POSIX time
        Returns:
            F_NGR_per_second     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        raise NotImplementedError

    def create_xarray(self, grids_dict: dict, data_tuple: np.array) -> xr:
        """Function to create an xarray from the data tuple and grid dict
        Args:
          data_tuple: np.array of the data (F_NGR_per_second_data)
          grids_dict: containing ranges and grids of x, y, t dimension
        Returns:
          xr     an xarray containing both the grid and data
        """
        # make a xarray object out of it
        return xr.Dataset(
            dict(
                F_NGR_per_second=(["time", "lat", "lon"], data_tuple),
            ),
            coords=dict(
                lon=grids_dict["x_grid"],
                lat=grids_dict["y_grid"],
                time=np.round(np.array(grids_dict["t_grid"]) * 1000, 0).astype("datetime64[ms]"),
            ),
        )

    def map_analytical_function_over_area(self, grids_dict: dict) -> np.array:
        """Function to map the analytical function over an area with the spatial states and grid_dict times.
        Args:
          grids_dict: containing grids of x, y, t dimension
        Returns:
          data_tuple     containing the data in tuple format as numpy array (not yet in xarray form)
        """

        # Step 1: Create the meshgrid numpy matrices for each coordinate
        LAT, TIMES, LON = np.meshgrid(
            grids_dict["y_grid"], grids_dict["t_grid"], grids_dict["x_grid"]
        )

        # Step 2: Feed the arrays into the solar radiation function and return the np.array
        F_NGR_per_second_data = self.F_NGR_per_second_analytical(lon=LON, lat=LAT, posix_time=TIMES)

        return F_NGR_per_second_data

    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> xr:
        """Function to get the data at a specific point.
        Args:
          spatio_temporal_point: SpatioTemporalPoint in the respective used coordinate system geospherical or unitless
        Returns:
          xr object that is then processed by the respective data source for its purpose
        """
        return self.F_NGR_per_second(spatio_temporal_point.to_spatio_temporal_casadi_input())

    def set_casadi_function(self):
        """Creates the symbolic computation graph for the full casadi function."""
        # Step 2: Set-Up the casadi graph for the full calculation
        sym_time = ca.MX.sym("time")  # in posix
        sym_lon_degree = ca.MX.sym("lon")  # in deg
        sym_lat_degree = ca.MX.sym("lat")  # in deg

        sym_NGR_per_second = self.F_NGR_per_second_analytical(
            lon=sym_lon_degree, lat=sym_lat_degree, posix_time=sym_time
        )

        # Set up the full function
        self.F_NGR_per_second = ca.Function(
            "d_biomass_dt_in_seconds",
            [ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree)],
            [sym_NGR_per_second],
        )


class SeaweedGrowthGEOMAR(SeaweedGrowthSource, AnalyticalSource):
    """Seaweed Growth based on the model from the paper below with various simplifications for our use-case (see Notion).
    Wu, Jiajun, David P. Keller, and Andreas Oschlies.
    "Carbon Dioxide Removal via Macroalgae Open-ocean Mariculture and Sinking: An Earth System Modeling Study."
     Earth System Dynamics Discussions (2022): 1-52.

     The nutrient concentrations of phosphate and nitrate as well as temperatures are taken
     from 2021 monthly averaged nutrient data from Copernicus.

     Example config: {
        'field': 'SeaweedGrowth',
        'source':'GEOMAR',
        'source_settings':{
            'filepath': './ocean_navigation_simulator/package_data/nutrients/'}  # './data/nutrients/2022_monthly_nutrients_and_temp.nc'
        }

     """

    def __init__(self, source_config_dict: Dict):
        """Dictionary with the three top level keys:
         'field' the kind of field the should be created, here SeaweedGrowth
         'source' = 'GEOMAR' for instantiating this source
         'source_settings':{
            'filepath': './data/2021_monthly_nutrients_and_temp.nc'
            'solar_source': pointer to solar source. Needs to be instantiated before this source.
        }
        """
        # Adding things that are required for the AnalyticalSource to the dictionary
        if 'x_domain' not in source_config_dict["source_settings"]:
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
        # TODO: this can be precomputed and saved as a .nc file to speed up the initialization
        nc_files = glob.glob(self.source_config_dict["source_settings"]["filepath"] + "*.nc")
        nc_files = sorted(nc_files, key=lambda x: xr.open_dataset(x).time[0].values)

        DataArray = xr.open_mfdataset(nc_files)

        DataArray = DataArray.rename({"latitude": "lat", "longitude": "lon"})
        DataArray = DataArray.assign(
            R_growth_wo_Irradiance=compute_R_growth_without_irradiance(
                DataArray["Temperature"], DataArray["no3"], DataArray["po4"]
            )
        )
        DataArray = DataArray.assign(R_resp=compute_R_resp(DataArray["Temperature"]))
        # Just to conserve RAM
        DataArray = DataArray.drop(["Temperature", "no3", "po4"])

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
        data_out = self.F_NGR_per_second(ca.DM([TIMES.flatten(), LAT.flatten(), LON.flatten()]))

        # LAT, LON = np.meshgrid(grids_dict["y_grid"], grids_dict["x_grid"])

        # LON, LAT = np.where((LON >= -79.9) & (LON <= -79.5), 1, 0), np.where(
        #     (LAT >= -14.9) & (LAT <= -14.5), 1, 0
        # )
        # data = np.multiply(LON.T, LAT.T)

        # T = grids_dict["t_grid"].shape[0]

        # data = np.repeat(data[np.newaxis, :, :], T, axis=0)
        return np.array(data_out).reshape(LAT.shape)


    def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> float:
        """Function to get the data at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          float of the net_growth_rate per second
        """
        return self.F_NGR_per_second(spatio_temporal_point.to_spatio_temporal_casadi_input())

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


class SeaweedGrowthCircles(SeaweedGrowthAnalytical):
    """Seaweed growth source with multiple circles of growth with different growth rates.
    The variables are specifeid in the source_settings.
     source_settings:
        cirles: [[-0.5, 1, 1]] # [x, y, r]
        NGF_in_time_units: [1] # [NGF]
    """

    def F_NGR_per_second_analytical(
            self,
            lon: Union[float, np.array],
            lat: Union[float, np.array],
            posix_time: Union[float, np.array],
    ) -> Union[float, np.array]:
        """Calculating the current in longitudinal direction at a specific lat, lon point at posix_time.
        Note: this can be used for floats as input or np.arrays which are all the same shape.
        Args:
            lon: longitude in degree
            lat: latitude in degree
            posix_time: POSIX time
        Returns:
            F_NGR_per_second     data as numpy array (not yet in xarray form) in 3D Matrix Time x Lat x Lon
        """
        # empty array to store the NGF
        NGF = 0
        # loop over all circles
        for circle, NGF_in_time_unit in zip(self.source_config_dict["source_settings"]['cirles'],
                                            self.source_config_dict["source_settings"]['NGF_in_time_units']):
            # Step 1: Create the meshgrid numpy matrices for each coordinate
            dx = lon - circle[0]
            dy = lat - circle[1]
            r = circle[2]
            signed_distance = np.sqrt(dx ** 2 + dy ** 2) - r
            # Step 2: Create the NGF and add it up
            NGF += (signed_distance < 0) * NGF_in_time_unit

        return NGF


class SeaweedGrowthCali(SeaweedGrowthGEOMAR):
    """Seaweed Growth based on the model from the paper below with various simplifications for our use-case (see Notion).
    Set 'source' = 'California' for instantiating this source in the SeaweedGrowthField
    Wu, Jiajun, David P. Keller, and Andreas Oschlies.
    "Carbon Dioxide Removal via Macroalgae Open-ocean Mariculture and Sinking: An Earth System Modeling Study."
     Earth System Dynamics Discussions (2022): 1-52.
     """

    def __init__(self, source_config_dict: Dict):
        """Dictionary with the three top level keys:
         'field' the kind of field the should be created, here SeaweedGrowth
         'source' = 'California' for instantiating this source
         'source_settings':{
            'filepath': './data/2021_monthly_nutrients_and_temp.nc'
            'solar_func_ca': pointer to the casadi function from the solar module to use here.
            'max_growth': 0.2
            'respiration_rate': 0.01
        }
        """
        # Adding things that are required for the AnalyticalSource to the dictionary
        source_config_dict = self.add_default_domains(source_config_dict)
        super().__init__(source_config_dict)

    def get_growth_and_resp_data_array_from_file(self) -> xr:
        """Helper function to open the dataset which contains the nutrient growth factor from 0-1."""
        # check if filepath is a folder using os.path.isdir
        if os.path.isdir(self.source_config_dict["source_settings"]["filepath"]):
            nc_files = glob.glob(self.source_config_dict["source_settings"]["filepath"] + "*.nc")
            nc_files = sorted(nc_files, key=lambda x: xr.open_dataset(x).time[0].values)
        else: # it is a single file
            nc_files = [self.source_config_dict["source_settings"]["filepath"]]

        DataArray = xr.open_mfdataset(nc_files)

        DataArray = DataArray.assign(
            R_growth_wo_Irradiance=
            self.source_config_dict["source_settings"]["max_growth"] * DataArray['static_growth_map']
        )

        return DataArray

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
        r_resp = self.source_config_dict["source_settings"]["respiration_rate"]

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
        ) * sym_f_Irradiance - r_resp / (24 * 3600)
        # Set up the full function
        self.F_NGR_per_second = ca.Function(
            "d_biomass_dt_in_seconds",
            [ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree)],
            [sym_NGR_per_second],
        )

    @staticmethod
    def add_default_domains(source_config_dict: Dict) -> Dict:
        """Helper Function to make it work smoothly with the AnalyticalSource class."""
        source_config_dict["source_settings"]["x_domain"] = [-124, -121.5]
        source_config_dict["source_settings"]["y_domain"] = [36, 38.98]
        source_config_dict["source_settings"]["temporal_domain"] = [
            datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        ]
        source_config_dict["source_settings"]["spatial_resolution"] = 0.03
        source_config_dict["source_settings"]["temporal_resolution"] = 3600

        return source_config_dict
