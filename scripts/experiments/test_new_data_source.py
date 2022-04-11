from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
import numpy as np
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

#%% Create the ocean Field
ocean_field = OceanCurrentField(hindcast_source_dict=source_dict)
#%% Use it
import datetime
t_0 = datetime.datetime.now() + datetime.timedelta(hours=10)
# t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=1)]
x_interval=[-82, -80]
y_interval=[24, 26]
x_0 = [-81.5, 23.5, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [-80, 24.2]

#%%
vec_point = ocean_field.get_forecast(point=x_T, time=t_0)
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




#%% Other stuff which is not shareable yet
#%% Solar irradiance Test
source_dict = {'field': 'SolarIrradiance',
               'subset_time_buffer_in_s': 4000,
               'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*5*12}}
source_dict['source'] = 'analytical'
import datetime
import numpy as np
source_dict['source_settings'] = {
                       'boundary_buffers': [0.2, 0.2],
                       'spatial_domain': [np.array([-90, 24]), np.array([-80, 26])],
                       'temporal_domain': [datetime.datetime.now().timestamp(), datetime.datetime.now().timestamp() + 3600*24*2],
                       'spatial_resolution': 0.1,
                       'temporal_resolution': 3600,
                   }
#%%
source_dict = {'field': 'OceanCurrents',
               'subset_time_buffer_in_s': 4000,
               'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*5*12}}

#% add stuff for analytical
source_dict['source'] = 'analytical'
source_dict['source_settings'] = {
                       'name': 'PeriodicDoubleGyre',
                       'boundary_buffers': [0.2, 0.2],
                       'spatial_domain': [np.array([-0.1, -0.1]), np.array([2.1, 1.1])],
                       'temporal_domain': [-10, 1000],
                       'spatial_resolution': 0.05,
                       'temporal_resolution': 10,
                       'v_amplitude': 2,
                       'epsilon_sep': 0.2,
                       'period_time': 10
                   }
#%%
import ocean_navigation_simulator.env.data_sources.SolarIrradianceField as SolarIrradianceField
solar_field = SolarIrradianceField.SolarIrradianceField(hindcast_source_dict=source_dict)
#%%
import datetime
t_0 = datetime.datetime.now() + datetime.timedelta(hours=10)
# t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=1)]
x_interval=[-82, -80]
y_interval=[24, 26]
x_0 = [-81.5, 23.5, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [-80, 24.2]
vec_point = solar_field.get_forecast(point=x_T, time=t_0)
print(vec_point)
#%%
area_xarray = solar_field.get_forecast_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)
# area_xarray = ocean_field.get_ground_truth_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)

#%% Test if casadi works here
ocean_field.hindcast_data_source.update_casadi_dynamics(x_0)
#%%
ocean_field.hindcast_data_source.casadi_grid_dict
#%%
ocean_field.hindcast_data_source.check_for_casadi_dynamics_update(x_0)
#%%
forecast_file_config_dict = {'source': 'forecast_files',
               'subset_time_buffer_in_s': 4000,
               'source_settings': {
                   'folder': "data/forecast_test/"# "data/cop_fmrc/"
               }}
#%% Hindcast file dict
hindcast_file_config_dict = {'source': 'hindcast_files',
               'subset_time_buffer_in_s': 4000,
               'source_settings': {
                   'folder': "data/single_day_hindcasts/"}, #"data/hindcast_test/"
               }
#%%
source_dict['source'] = 'opendap'
source_dict['source_settings'] = {
                   'service': 'copernicus',
                   'currents': 'total', # if we want to take the normal uo, vo currents or 'total' for tide, normal added
                   'USERNAME': 'mmariuswiggert', 'PASSWORD': 'tamku3-qetroR-guwneq',
                   # 'DATASET_ID': 'global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh',
                   'DATASET_ID': 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i'}
#%%
ocean_field = OceanCurrentField(hindcast_source_dict=source_dict)
#%%
empty_arr = ocean_field.hindcast_data_source.DataArray.sel(lat=slice(-90, -85))
#%%
import datetime
t_0 = datetime.datetime(2021, 11, 23, 12, 10, 10, tzinfo=datetime.timezone.utc)
# t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=1)]
x_interval=[-82, -80]
y_interval=[24, 26]
x_0 = [-81.5, 23.5, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [-80, 24.2]
#%% for double gyre testing
import datetime
t_0 = datetime.datetime.fromtimestamp(10, datetime.timezone.utc)
# t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(seconds=20)]
x_interval=[0, 2]
y_interval=[0, 1]
x_0 = [1.5, 0.5, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [0.1, 1.]
#%%
vec_point = ocean_field.get_forecast(point=x_T, time=t_0)
print(vec_point)
vec_point = ocean_field.get_ground_truth(point=x_T, time=t_0)
print(vec_point)
#%%
area_xarray = ocean_field.get_forecast_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)
# area_xarray = ocean_field.get_ground_truth_area(x_interval=x_interval, y_interval=y_interval, t_interval=t_interval)

#%% Test if casadi works here
ocean_field.hindcast_data_source.update_casadi_dynamics(x_0)
#%%
ocean_field.hindcast_data_source.casadi_grid_dict
#%%
ocean_field.hindcast_data_source.check_for_casadi_dynamics_update(x_0)
#%%
ocean_field.hindcast_data_source.v_curr_func