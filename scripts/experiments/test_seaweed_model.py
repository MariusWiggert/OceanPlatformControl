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
    lon=units.Distance(deg=-50.7),
    lat=units.Distance(deg=-44.2),
    seaweed_mass=units.Mass(kg=100),
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
#%%
from ocean_navigation_simulator.env.data_sources.SeaweedGrowthField import SeaweedGrowthField
#% initialize the solar model
from ocean_navigation_simulator.env.data_sources.SolarIrradiance.SolarIrradianceSource import AnalyticalSolarIrradiance
solar_source_dict['casadi_cache_settings'] = sim_cache_dict
solar_source = AnalyticalSolarIrradiance(solar_source_dict)
solar_source.update_casadi_dynamics(platform_state)
#%% Set Seaweed Model dict
seaweed_source_dict = {
    'field': 'SeaweedGrowth',
    'source': 'GEOMAR',
    'source_settings': {
        'filepath': './data/Nutrients/2021_monthly_nutrients_and_temp.nc',
        'solar_source': solar_source,
    }
}
#%% Instantiate the Source within the Field object -> this takes ≈40 seconds
seaweed_field = SeaweedGrowthField(sim_cache_dict=sim_cache_dict, hindcast_source_dict=seaweed_source_dict)
#%% Did it work?
to_plot_time = datetime.datetime(year=2021, month=11, day=20, hour=19, minute=0, second=0, tzinfo=datetime.timezone.utc)
seaweed_field.hindcast_data_source.plot_R_growth_wo_Irradiance(to_plot_time)
#%% The growth at the platform_state if the sun shines as at that time for 12h
seaweed_field.hindcast_data_source.F_NGR_per_second(platform_state.to_spatio_temporal_casadi_input())*12*3600

#%% get the data over an area (e.g. for hj_reachability or when we implement caching)
x_interval = [-150, 150]
y_interval = [-60, 60]
t_interval = [datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2021, 1, 1, 23, 0, 0, tzinfo=datetime.timezone.utc)]
#%% get a subset
subset = seaweed_field.hindcast_data_source.get_data_over_area(
    x_interval, y_interval, t_interval, spatial_resolution=5, temporal_resolution=3600*6)
#%% plot the subset
import matplotlib.pyplot as plt
subset['F_NGR_per_second'].isel(time=0).plot()
plt.show()

