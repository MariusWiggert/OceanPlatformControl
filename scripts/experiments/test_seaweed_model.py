import datetime
import numpy as np
import xarray as xr
import time

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Platform import Platform, PlatformState
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.problem import Problem

from ocean_navigation_simulator.env.utils import units

platform_state = PlatformState(
    date_time=datetime.datetime(year=2021, month=11, day=20, hour=12, minute=0, second=0, tzinfo=datetime.timezone.utc),
    lon=units.Distance(deg=-83.69599051807417),
    lat=units.Distance(deg=27.214803181574762),
    seaweed_mass=units.Mass(kg=0),
    battery_charge=units.Energy(watt_hours=100)
)
#%%
sim_cache_dict = {'deg_around_x_t': 1, 'time_around_x_t': 3600 * 24 * 3}
platform_dict = {
    'battery_cap_in_wh': 400.0,
    'u_max_in_mps': 0.1,
    'motor_efficiency': 1.0,
    'solar_panel_size': 0.5,
    'solar_efficiency': 0.2,
    'drag_factor': 675,
    'dt_in_s': 600,
}

solar_source_dict = {
    'field': 'SolarIrradiance',
    'source': 'analytical',
    'source_settings': {
        'boundary_buffers': [0.2, 0.2],
        'x_domain': [-180, 180],
        'y_domain': [-90, 90],
        'temporal_domain': [datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
                            datetime.datetime(2023, 1, 10, 0, 0, 0, tzinfo=datetime.timezone.utc)],
        'spatial_resolution': 0.1,
        'temporal_resolution': 3600,
    }
}
#%% initialize the solar model
from ocean_navigation_simulator.env.data_sources.SolarIrradiance.SolarIrradianceSource import AnalyticalSolarIrradiance
solar_source_dict['casadi_cache_settings'] = sim_cache_dict
solar_source = AnalyticalSolarIrradiance(solar_source_dict)
solar_source.update_casadi_dynamics(platform_state)
#%% Set Seaweed Model dict
seaweed_source_dict = {
    'field': 'SeaweedGrowth',
    'source': 'GEOMAR_paper',
    'source_settings': {
        'filepath': './data/Nutrients/2021_monthly_nutrients_and_temp.nc',
        'solar_func': solar_source.solar_rad_casadi,
    }
}
# %% Initialize the Controller
# Open the nutrient dataset and calculate the relevant quantities from it
DataArray = xr.open_dataset(seaweed_source_dict['source_settings']['filepath'])
#%%
from ocean_navigation_simulator.env.data_sources.SeaweedGrowth.SeaweedFunction import *
start = time.time()
DataArray = DataArray.assign(R_growth_wo_Irradiance=compute_R_growth_without_irradiance(DataArray['Temperature'], DataArray['no3'], DataArray['po4']))
DataArray = DataArray.assign(R_resp=compute_R_resp(DataArray['Temperature']))
DataArray = DataArray.drop(['Temperature', 'no3', 'po4'])   # Just to conserve RAM
print(f'Calculating of nutrient derived data array takes: {time.time() - start:.2f}s')
#%% get the Net Growth Rate for a point (or a region)
DataArray = DataArray.assign(NGR_wo_Irradiance=DataArray['R_growth_wo_Irradiance'] - DataArray['R_resp'])
#%%
import matplotlib.pyplot as plt
DataArray['NGR_wo_Irradiance'].isel(time=0).plot()
plt.show()
#%%
seaweed_mass = 100
#%%
# Equation is:
# Net_Growth_Rate = R_growth_wo_Irradiance (closest) * Irradiance_Factor - R_resp (closest)
# dBiomass_dt = Mass*Net_Growth_Rate

