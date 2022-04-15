import datetime
import matplotlib.pyplot as plt
import numpy as np
from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
import ocean_navigation_simulator.env.data_sources.SolarIrradianceField as SolarIrradianceField
#%% Solar irradiance Test
# Step 1: create the specification dict
source_dict = {'field': 'SolarIrradiance',
               'subset_time_buffer_in_s': 4000,
               'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*1*24}}
source_dict['source'] = 'analytical'
source_dict['source_settings'] = {
                       'boundary_buffers': [0.2, 0.2],
                       'x_domain': [-180, 180],
                        'y_domain': [-90, 90],
                       'temporal_domain': [datetime.datetime(2020, 1, 1, 0, 0, 0),
                                           datetime.datetime(2023, 1, 10, 0, 0, 0)],
                       'spatial_resolution': 0.1,
                       'temporal_resolution': 3600,
                   }
#%% Step 2: Instantiate the field
solar_field = SolarIrradianceField.SolarIrradianceField(hindcast_source_dict=source_dict)
# Test settings to use it
t_0 = datetime.datetime(2022, 4, 11, 0, 0, 0, tzinfo=datetime.timezone.utc)
# t_0 = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(hours=10)
# t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=1)]
x_interval = [-122, -120]
y_interval = [35, 37]
x_0 = [-120, 35, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [-122, 37]
#%% check it for a point
vec_point = solar_field.get_forecast(point=x_T, time=t_0)
print(vec_point)
#%% check it for a spatio-temporal area
area_xarray = solar_field.get_ground_truth_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)
#%% check if it makes sense over Berkeley
area_xarray.isel(lat=0, lon=0)['solar_irradiance'].plot()
plt.show()
#%% Test if casadi works here
solar_field.hindcast_data_source.update_casadi_dynamics(x_0)
#%%
solar_field.hindcast_data_source.casadi_grid_dict
#%%
solar_field.hindcast_data_source.check_for_casadi_dynamics_update(x_0)
#%% the casadi function for use in simulation
solar_field.hindcast_data_source.solar_rad_casadi




#%% Create the source dict for the ocean currents
source_dict = {'field': 'OceanCurrents',
               'subset_time_buffer_in_s': 4000,
               'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*5*12}}
source_dict['source'] = 'opendap'
source_dict['source_settings'] = {
                    'service': 'copernicus',
                    'currents': 'total', # if we want to take the normal uo, vo currents or 'total' for tide, normal added
                    'USERNAME': 'mmariuswiggert', 'PASSWORD': 'tamku3-qetroR-guwneq',
                    # 'DATASET_ID': 'global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh',
                    'DATASET_ID': 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i'}

#%% forecast file options (if in local folder)
forecast_file_config_dict = {'source': 'forecast_files',
               'subset_time_buffer_in_s': 4000,
               'source_settings': {
                   'folder': "data/forecast_test/"# "data/cop_fmrc/"
               }}
#%% Hindcast file dict (if in local folder)
hindcast_file_config_dict = {'source': 'hindcast_files',
               'subset_time_buffer_in_s': 4000,
               'source_settings': {
                   'folder': "data/single_day_hindcasts/"}, #"data/hindcast_test/"
               }

#%% Create the ocean Field
ocean_field = OceanCurrentField(hindcast_source_dict=source_dict)# | hindcast_file_config_dict)
#ocean_field = OceanCurrentField(hindcast_source_dict=source_dict | hindcast_file_config_dict)
#%%
ocean_field.hindcast_data_source.grid_dict
#%% Use it
#t_0 = datetime.datetime.fromtimestamp(1637600400) + datetime.timedelta(hours=10)
#print(t_0)
t_0 = datetime.datetime(2021, 12, 5, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=5)]
x_interval=[-82, -80]
y_interval=[24, 40]
x_0 = [-81.5, 23.5, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [-80, 24.2]

#%%
vec_point = ocean_field.get_forecast_area(x_interval=x_interval,y_interval=y_interval, t_interval=t_interval)
print(vec_point)
vec_point = ocean_field.get_ground_truth(point=x_T, time=t_0)
print(vec_point)
#%% Not working some weird error but you don't need it
# area_xarray = ocean_field.get_forecast_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)
# area_xarray = ocean_field.get_ground_truth_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)
#%% Passed to the platform is then the object at initialization
data_source_in_platform = ocean_field.hindcast_data_source
#%% Checks to run
data_source_in_platform.update_casadi_dynamics(x_0)
#%%
data_source_in_platform.check_for_casadi_dynamics_update(x_0)
#%% inside casadi dynamics
data_source_in_platform.u_curr_func
data_source_in_platform.v_curr_func




#%% Analytical Ocean Current Example
source_dict = {'field': 'OceanCurrents',
               'subset_time_buffer_in_s': 4000,
               'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 500}}
source_dict['source'] = 'analytical'
source_dict['source_settings'] = {
                       'name': 'PeriodicDoubleGyre',
                       'boundary_buffers': [0.2, 0.2],
                        'x_domain': [-0.1, 2.1],
                        'y_domain': [-0.1, 1.1],
                       'temporal_domain': [-10, 1000], # will be interpreted as POSIX timestamps
                       'spatial_resolution': 0.05,
                       'temporal_resolution': 10,
                       'v_amplitude': 1,
                       'epsilon_sep': 0.2,
                       'period_time': 10
                   }
#%% Create the ocean Field
ocean_field = OceanCurrentField(hindcast_source_dict=source_dict)
#%%
ocean_field.hindcast_data_source.viz_field()
#%% Use it
t_0 = datetime.datetime.fromtimestamp(10, datetime.timezone.utc)
# t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(seconds=20)]
x_interval=[0, 2]
y_interval=[0, 1]
x_0 = [1.5, 0.5, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [0.1, 1.]

# Why outside area issues and why everything zero?
#%%
vec_point = ocean_field.get_forecast(point=x_T, time=t_0)
print(vec_point)
vec_point = ocean_field.get_ground_truth(point=x_T, time=t_0)
print(vec_point)
#%% Not working some weird error but you don't need it
area_xarray = ocean_field.get_forecast_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)
# area_xarray = ocean_field.get_ground_truth_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)
#%% Passed to the platform is then the object at initialization
data_source_in_platform = ocean_field.hindcast_data_source
#%% Checks to run
data_source_in_platform.update_casadi_dynamics(x_0)
#%%
data_source_in_platform.check_for_casadi_dynamics_update(x_0)
#%% inside casadi dynamics
data_source_in_platform.u_curr_func
data_source_in_platform.v_curr_func
