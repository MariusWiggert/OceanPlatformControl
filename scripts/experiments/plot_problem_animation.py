import datetime
import matplotlib.pyplot as plt
import numpy as np

from ocean_navigation_simulator.env.PlatformState import SpatioTemporalPoint, PlatformState, SpatialPoint
from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
import ocean_navigation_simulator.env.data_sources.SolarIrradianceField as SolarIrradianceField
from ocean_navigation_simulator.env.utils import units
sim_cache_dict = {'deg_around_x_t': 1, 'time_around_x_t': 3600 * 24 * 1}
#% Analytical Ocean Current Example
ocean_source_dict = {
                'field': 'OceanCurrents',
                'source': 'analytical',
                'source_settings': {
                    'name': 'PeriodicDoubleGyre',
                       'boundary_buffers': [0.1, 0.1],
                        'x_domain': [-0.1, 2.1],
                        'y_domain': [-0.1, 1.1],
                       'temporal_domain': [0, 1000], # will be interpreted as POSIX timestamps
                       'spatial_resolution': 0.05,
                       'temporal_resolution': 5,
                       'v_amplitude': 1,
                       'epsilon_sep': 0.2,
                       'period_time': 50
                   }}
# Create the ocean Field
ocean_field = OceanCurrentField(hindcast_source_dict=ocean_source_dict,
                                sim_cache_dict=sim_cache_dict,
                                use_geographic_coordinate_system=False)
#%% visualize currents
ocean_field.forecast_data_source.plot_currents_at_time(
    time=10, x_interval=[-2, 10], y_interval=[-2, 10], return_ax=False, max_spatial_n=500)
#%%
#% Input to animate the currents
x_interval=[-2, 10]
y_interval=[-2, 10]
#%% test add_ax_func
def add_dot(ax, posix_time):
    ax.scatter(1,1, s=800, color='red')
#%%
ocean_field.hindcast_data_source.animate_currents(x_interval=x_interval, y_interval=y_interval,
                                                  t_interval=[0, 100], save_as_filename='full_test.gif',
                                                  html_render='safari', max_spatial_n=50, figsize=(24, 12),
                                                  add_ax_func=add_dot)