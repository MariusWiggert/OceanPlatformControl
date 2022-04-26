import datetime
import matplotlib.pyplot as plt
import numpy as np
from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
import ocean_navigation_simulator.env.data_sources.SolarIrradianceField as SolarIrradianceField
from ocean_navigation_simulator.env.utils import units

ocean_source_dict = {
    'field': 'OceanCurrents',
    'source': 'analytical',
    'source_settings': {
        'name': 'FixedCurrentHighwayField',
        'boundary_buffers': [0.2, 0.2],
        'x_domain': [0, 10],
        'y_domain': [0, 10],
        'temporal_domain': [0, 10],
        'spatial_resolution': 0.1,
        'temporal_resolution': 1,
        'y_range_highway': [4, 6],
        'U_cur': 2,
    },
}

sim_cache_dict = {'deg_around_x_t': 1, 'time_around_x_t': 100}

#%% Create the ocean Field
ocean_field = OceanCurrentField(hindcast_source_dict=ocean_source_dict, sim_cache_dict=sim_cache_dict)
#%%
from ocean_navigation_simulator.env.PlatformState import PlatformState
platform_state = PlatformState(
    date_time=datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc),
    lon=units.Distance(deg=2),
    lat=units.Distance(deg=2),
    seaweed_mass=units.Mass(kg=0),
    battery_charge=units.Energy(watt_hours=100)
)
#%%
ocean_field.hindcast_data_source.update_casadi_dynamics(platform_state)