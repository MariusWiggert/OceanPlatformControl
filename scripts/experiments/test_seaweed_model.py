import datetime
import numpy as np
import xarray as xr
import time

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Platform import Platform, PlatformState
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.problem import Problem
from ocean_navigation_simulator.env.data_sources.DataSources import DataSource

from ocean_navigation_simulator.env.utils import units

platform_state = PlatformState(
    date_time=datetime.datetime(year=2021, month=11, day=20, hour=19, minute=0, second=0, tzinfo=datetime.timezone.utc),
    lon=units.Distance(deg=-83.7),
    lat=units.Distance(deg=27.2),
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
DataArray = DataArray.rename({'latitude': 'lat', 'longitude': 'lon'})
#%%
from ocean_navigation_simulator.env.data_sources.SeaweedGrowth.SeaweedFunction import *
start = time.time()
DataArray = DataArray.assign(R_growth_wo_Irradiance=compute_R_growth_without_irradiance(DataArray['Temperature'], DataArray['no3'], DataArray['po4']))
DataArray = DataArray.assign(R_resp=compute_R_resp(DataArray['Temperature']))
DataArray = DataArray.drop(['Temperature', 'no3', 'po4'])   # Just to conserve RAM
print(f'Calculating of nutrient derived data array takes: {time.time() - start:.2f}s')
#%% For Gut checking calculate NGR without Irradiance and plot it
# Assuming respiration can dominate photosynthesis
NGR_wo_Irradiance = DataArray['R_growth_wo_Irradiance'] - DataArray['R_resp']
# Assuming the growth is never negative
NGR_wo_Irradiance = np.maximum(0, NGR_wo_Irradiance)
DataArray = DataArray.assign(NGR_wo_Irradiance=NGR_wo_Irradiance)
import matplotlib.pyplot as plt
DataArray['R_growth_wo_Irradiance'].isel(time=0).plot()
plt.show()
#%% In the SeaweedGrowthSource we calculate the specific NGR for a platform_state
# First get solar for platform_state
Irradiance_at_state = seaweed_source_dict['source_settings']['solar_func'](platform_state.to_spatio_temporal_casadi_input())
# Calculate solar factor
f_Im = irradianceFactor(Irradiance_at_state)
# put together with the rest to get NGR
NGR_at_state =
#%% get the interpolation functions
casadi_grid_dict = DataSource.get_grid_dict_from_xr(DataArray)
#%%
# Step 3: Set up the grid
grid = [
    posix_to_rel_seconds_in_year(units.get_posix_time_from_np64(DataArray.coords['time'].values)),
    DataArray.coords['lat'].values,
    DataArray.coords['lon'].values
]
# Set-up the casadi interpolation function
import casadi as ca
r_growth_wo_irradiance = ca.interpolant('r_growth_wo_irradiance', 'linear', grid, DataArray['R_growth_wo_Irradiance'].values.ravel(order='F'))
r_resp = ca.interpolant('r_resp', 'linear', grid, DataArray['R_resp'].values.ravel(order='F'))
#%% Test the interpolation functions if they work as expected
test_time = datetime.datetime(year=2021, month=11, day=20, hour=19, minute=0, second=0, tzinfo=datetime.timezone.utc).timestamp()
time_grid = np.array(test_time)
lat_grid = grid[1]
lon_grid = grid[2]
LAT, T, LON = np.meshgrid(lat_grid, time_grid, lon_grid)
# Stack them together to a new matrix
input_to_interpol = np.concatenate([T, LAT, LON], axis=0)
#%% For a point
r_growth_wo_irradiance([posix_to_rel_seconds_in_year(test_time), 40, -150])
#%% Feed into function
# Construct array to feed in!
NGR_over_globe = r_growth_wo_irradiance(ca.MX(input_to_interpol))
#%%
ca.MX(input_to_interpol)
#%%
ca.MX(LAT.squeeze())
#%% Need the interpolation functions...
# DataArray['R_growth_wo_Irradiance'].sel(time=np.datetime64(platform_state.date_time), method="nearest")
# Key Question: how can we run the interpolation function for months when input is posix time?

def posix_to_rel_seconds_in_year(posix_timestamp: float) -> float:
    """Helper function to map a posix_timestamp to it's relative seconds for the specific year (since 1st of January).
    This is needed because the interpolation function for the nutrients operates on relative timestamps as we take
    the average monthly nutrients for those as input.
    Args:
        posix_timestamp: a posix timestamp
    """
    # correction for extra long years because of Schaltjahre (makes it accurate 2020-2024, and outside a couple of days off)
    correction_seconds = 13 * 24 * 3600
    # Calculate the relative time of the year in seconds
    return np.mod(posix_timestamp, 365*24*3600) - correction_seconds

def posix_to_2021_year_posix(posix_timestamp: float) -> float:
    """Helper function to map a posix_timestamp from any year 2021 onwards to the month/day/time in 2021 posix.
    This is needed because the interpolation function for the nutrients operates on 2021 posix timestamps as we take
    the average monthly nutrients for those as input.
    Args:
        posix_timestamp: a posix timestamp
    """
    posix_2021_1_1_utc = 1609459200
    # Calculate the delta_2021 in seconds
    delta_2021 = posix_timestamp - posix_2021_1_1_utc
    if delta_2021 < 0:  # If negative -> Error because function only works for 2021
        raise ValueError("posix_to_2021_year_posix only implemented for inputs 2021 onwards")
    # divide with rest by the yearly amount of seconds to get the seconds relative to 1st of January
    relative_seconds_of_year = delta_2021 % (365*24*3600)
    # add this to 2021_1_1 to get the 2021 timestamp for the same month dat and time
    return posix_2021_1_1_utc + relative_seconds_of_year

# need to check extrapolation at the boundaries e.g. 1.1 or 31.12 should be constant...
#%% testing
out = posix_to_2021_year_posix(datetime.datetime(2024, 10, 1, 10, 0, tzinfo=datetime.timezone.utc).timestamp())
print(datetime.datetime.fromtimestamp(out, tz=datetime.timezone.utc))
#%%
DataArray['R_growth_wo_Irradiance'] =

#%%
seaweed_mass = 100
#%%
# Equation is:
# Net_Growth_Rate = R_growth_wo_Irradiance (closest) * Irradiance_Factor - R_resp (closest)
# dBiomass_dt = Mass*Net_Growth_Rate

