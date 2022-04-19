#%% imports
import datetime
import matplotlib.pyplot as plt
from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
import ocean_navigation_simulator.env.data_sources.OceanCurrentField as OceanCurrentField

# Step 1: create the specification dict
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.utils import units
from scripts.experiments.class_gp import OceanCurrentGP

#%% create the source dict for the ocean currents
source_dict = {'field': 'OceanCurrent',
               'subset_time_buffer_in_s': 4000,
               'casadi_cache_settings': {'deg_around_x_t': 2, 'time_around_x_t': 3600*5*12},
               }#'source': 'hindcast_files'} #not sure hindcast here
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

#%% Step 2: Instantiate the field
# current_field = OceanCurrentField.OceanCurrentField(hindcast_source_dict = source_dict | hindcast_file_config_dict,
#                                                     forecast_source_dict= source_dict | forecast_file_config_dict)
current_field = OceanCurrentField.OceanCurrentField(hindcast_source_dict = source_dict | forecast_file_config_dict, forecast_source_dict=source_dict|forecast_file_config_dict)
# Test settings to use it
t_0 = datetime.datetime.now() + datetime.timedelta(hours=10)

#test gulf of Mexico:
# long: -95.053711  -84.111328
# lat:  23.497506    28.124313

t_interval = [datetime.datetime(2021,11,22), datetime.datetime(2021,12,26)]
x_interval=[-95.053711, -84.111328]
y_interval=[23.497506, 28.124313]

x_0 = [-81.5, 23.5, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [-80, 24.2]


#%%
vec_point = current_field.get_forecast_area(x_interval, y_interval, t_interval)
print(vec_point)
#%% New part
from scripts.experiments.class_gp import OceanCurrentGP
model = OceanCurrentGP(current_field)
x,y = units.Distance(m=0.0), units.Distance(m=0.0)
t_0 = datetime.datetime.fromtimestamp(10, datetime.timezone.utc)
t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
ocean_current_vector = OceanCurrentVector(1.0, 1.0)

#%%
vec_point = current_field.get_forecast(point=[x.m, y.m], time=t_0)
print('forecast: ', vec_point)
pre_measurement = model.query(x, y, t_0)
model.observe(x, y, t_0, ocean_current_vector)
post_measurement = model.query(x, y, t_0)
print('results:',pre_measurement, post_measurement)



ocean_field = OceanCurrentField(hindcast_source_dict=source_dict | hindcast_file_config_dict)

ocean_field.hindcast_data_source.grid_dict