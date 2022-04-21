import time

import casadi as ca
import datetime
from typing import List, NamedTuple, Sequence, AnyStr, Optional, Tuple, Union, Dict
import numpy as np
import xarray as xr
from ocean_navigation_simulator.env.data_sources.DataSources import DataSource, XarraySource
from ocean_navigation_simulator.env.data_sources.SolarIrradiance.SolarIrradianceSource import *
from ocean_navigation_simulator.env.data_sources.SeaweedGrowth.SeaweedFunction import *
from ocean_navigation_simulator.env.data_sources.SeaweedGrowth.SeaweedNutrientsSource import *

# source_dict = {'field': 'SeaweedGrowth',
#                 'source': 'file',
#                'subset_time_buffer_in_s': 3600*24*31,
#                'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*31},
#                'source_settings': {'filepath': 'data'}
#             }

# solar_dict = {'field': 'SolarIrradiance',
#                'subset_time_buffer_in_s': 4000,
#                'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*1*24},
#                'source': 'analytical',
#                'source_settings': {
#                        'boundary_buffers': [0.2, 0.2],
#                        'x_domain': [-180, 180],
#                         'y_domain': [-90, 90],
#                        'temporal_domain': [datetime.datetime(2020, 1, 1, 0, 0, 0),
#                                            datetime.datetime(2023, 1, 10, 0, 0, 0)],
#                        'spatial_resolution': 0.1,
#                        'temporal_resolution': 3600,
#                    }
# }
# source_dict['solar_dict'] = solar_dict


class SeaweedGrowthSource(DataSource):
    """Base class for the Seaweed Growth data sources.
    Note: It requires input from the Solar Source in W/m^2 and
    """

    def initialize_casadi_functions(self, grid: List[List[float]], array: xr) -> None:
        """DataSource specific function to initialize the casadi functions needed.
        Note: the input to the casadi function needs to be an array of the form np.array([posix time, lat, lon])
        Args:
          grid:     list of the 3 grids [time, y_grid, x_grid] for the xr data
          array:    xarray object containing the sub-setted data for the next cached round
        """
        self.r_growth = ca.interpolant('r_growth', 'linear', grid, array['R_growth'].values.ravel(order='F'))
        self.r_resp = ca.interpolant('r_resp', 'linear', grid, array['R_resp'].values.ravel(order='F'))


class SeaweedGrowthXarray(SeaweedGrowthSource, XarraySource):
    """Seaweed Growth based on the model from the paper below with various simplifications for our use-case (see Notion).
    Set 'source' = 'GEOMAR_paper' for instantiating this source in the SeaweedGrowthField
    Wu, Jiajun, David P. Keller, and Andreas Oschlies.
    "Carbon Dioxide Removal via Macroalgae Open-ocean Mariculture and Sinking: An Earth System Modeling Study."
     Earth System Dynamics Discussions (2022): 1-52.

     The nutrient concentrations of phosphate and nitrate as well as temperatures are taken
     from 2021 monthly averaged nutrient data from Copernicus."""

    def __init__(self, source_config_dict: Dict):
        """ Dictionary with the three top level keys:
             'field' the kind of field the should be created, here SeaweedGrowth
             'source' = 'GEOMAR_paper' for instantiating this source
             'source_settings':{
                'filepath': './data/2021_monthly_nutrients_and_temp.nc'
                'solar_func': pointer to the casadi function from the solar module to use here.
            }
        """
        super().__init__(source_config_dict)
        # Initialize variables used to hold casadi functions.
        self.net_growth_rate, self.r_growth, self.r_resp = [None]*3
        self.solar_rad_casadi = source_config_dict['solar_source'].solar_rad_casadi

        # Open the nutrient dataset and calculate the relevant quantities from it
        # TODO: we can actually just pre-compute it and just load the nc file for those two values.
        self.DataArray = xr.open_dataset(source_config_dict['source_settings']['filepath'])
        start = time.time()
        self.DataArray = self.DataArray.assign(R_growth=compute_R_growth_without_irradiance(self.DataArray['Temperature'], self.DataArray['no3'], self.DataArray['po4']))
        self.DataArray = self.DataArray.assign(R_resp=compute_R_resp(self.DataArray['Temperature']))
        print(f'Calculating of nutrient derived data array takes: {time.time() - start:.2f}s')

    def get_data_at_point(self, point: List[float], time: datetime.datetime) -> float:
        """Function to get the data at a specific point.
        Args:
          point: Point in the respective used coordinate system e.g. [lon, lat] for geospherical or unitless for examples
          time: absolute datetime object
        Returns:
          xr object that is then processed by the respective data source for its purpose
          """
        # add seaweed as an optional parameter
        # ngr = super().get_data_at_point(point, time)
        #
        # # now use state to calculate dBiomass_dt
        # return np.multiply(seaweed_mass, np.subtract(ngr, R_erosion * seaweed_mass))

    # def get_data_over_area(self, x_interval: List[float], y_interval: List[float],
    #                        t_interval: List[datetime.datetime],
    #                        spatial_resolution: Optional[float] = None,
    #                        temporal_resolution: Optional[float] = None) -> xr:
    #     # add seaweed as an optional parameter
    #     ngr = super().get_data_over_area(x_interval=x_interval,
    #                                     t_interval=t_interval,
    #                                     spatial_resolution=spatial_resolution,
    #                                     temporal_resolution=temporal_resolution)
    #
    #     # now use state to calculate dBiomass_dt
    #     return np.multiply(seaweed_mass, np.subtract(ngr, R_erosion * seaweed_mass))

