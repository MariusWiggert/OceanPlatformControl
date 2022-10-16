import datetime
import matplotlib.pyplot as plt

from ocean_navigation_simulator.environment.PlatformState import SpatioTemporalPoint, PlatformState, SpatialPoint
import ocean_navigation_simulator.data_sources.SolarIrradianceField as SolarIrradianceField
from ocean_navigation_simulator.utils import units
casadi_cache_dict = {'deg_around_x_t': 1, 'time_around_x_t': 3600 * 24 * 1}
#% Solar irradiance Test
# Step 1: create the specification dict
source_dict = {'field': 'SolarIrradiance',
               'source': 'analytical_wo_caching', # can also be analytical_w_caching
               'source_settings': {
                       'boundary_buffers': [0.2, 0.2],
                       'x_domain': [-180, 180],
                        'y_domain': [-90, 90],
                       'temporal_domain': [datetime.datetime(2020, 1, 1, 0, 0, 0),
                                           datetime.datetime(2023, 1, 10, 0, 0, 0)],
                       'spatial_resolution': 0.1,
                       'temporal_resolution': 3600,
                   }}
#% Step 2: Instantiate the field
solar_field = SolarIrradianceField.SolarIrradianceField(hindcast_source_dict=source_dict,
                                                        casadi_cache_dict=casadi_cache_dict)
#%% Test settings to use it
t_0 = datetime.datetime(2022, 4, 11, 0, 0, 0, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=3)]
x_interval = [-170, +170]
y_interval = [-80, +80]
x_0 = PlatformState(lon=units.Distance(deg=-120), lat=units.Distance(deg=30), date_time=t_0)
x_T = SpatialPoint(lon=units.Distance(deg=-122), lat=units.Distance(deg=37))
#%% Get solar irradiance at point
radiation = solar_field.hindcast_data_source.get_data_at_point(spatio_temporal_point=x_0)
print("Solar Irradiance at x_0 in W/M^2 is :", radiation)
#%% Plot it at time over area
solar_field.hindcast_data_source.plot_data_at_time_over_area(time=t_0, x_interval=x_interval, y_interval=y_interval,
                                                             spatial_resolution=5)
#%% Animate it over time
solar_field.hindcast_data_source.animate_data(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval,
                                              spatial_resolution=5, temporal_resolution=3600 * 3, output="solar_test_animation.mp4")
#%% check it for a spatio-temporal area
area_xarray = solar_field.get_ground_truth_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval,
                                                spatial_resolution=5, temporal_resolution=3600*3)
#%% check if it for a specific location isel selects rows and columns not actual degrees
area_xarray.isel(lat=15, lon=15)['solar_irradiance'].plot()
plt.show()
#%% Test if casadi works here (only if the solar_field is a caching one)
solar_field.hindcast_data_source.update_casadi_dynamics(x_0)
#%% Further tests to check inside is working
# solar_field.hindcast_data_source.casadi_grid_dict
# solar_field.hindcast_data_source.check_for_casadi_dynamics_update(x_0)
# #% the casadi function for use in simulation
# solar_field.hindcast_data_source.solar_rad_casadi