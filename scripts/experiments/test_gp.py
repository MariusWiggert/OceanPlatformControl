#%% imports
import datetime
import matplotlib.pyplot as plt
from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
import ocean_navigation_simulator.env.data_sources.OceanCurrentField as OceanCurrentField

# Step 1: create the specification dict
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.utils import units
from scripts.experiments.class_gp import OceanCurrentGP
import pandas as pd
import numpy as np

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

#%% sim_cache dict
sim_cache = {}

#%% Step 2: Instantiate the field
# current_field = OceanCurrentField.OceanCurrentField(hindcast_source_dict = source_dict | hindcast_file_config_dict,
#                                                     forecast_source_dict= source_dict | forecast_file_config_dict)
current_field = OceanCurrentField.OceanCurrentField(sim_cache,
                                                    hindcast_source_dict=source_dict|hindcast_file_config_dict,
                                                    forecast_source_dict=source_dict|forecast_file_config_dict)
# Test settings to use it
t_0 = datetime.datetime.now() + datetime.timedelta(hours=10)

#test gulf of Mexico:
# long: -95.053711  -84.111328
# lat:  23.497506    28.124313

#t_interval = [datetime.datetime(2021,11,28), datetime.datetime(2021,12,12)]
t_0 = datetime.datetime(2021, 11, 26, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=25)]
x_interval=[-95.053711, -89.111328]#[-95.053711, -84.111328]
y_interval=[23.497506, 25.124313]#[23.497506, 28.124313]

x_0 = [-81.5, 23.5, 1, t_0.timestamp()]  # lon, lat, battery
x_T = [-80, 24.2]


#%%
vec_point_forecast = current_field.get_forecast_area(x_interval, y_interval, t_interval)
print(vec_point_forecast)
#%%
vec_point_hindcast = current_field.get_ground_truth_area(x_interval, y_interval, t_interval)
print(vec_point_hindcast)
#%% Convert the data into arrays for training and testing
array_forecast = vec_point_forecast.to_array().transpose("time","lat","lon","variable")
array_hindcast = vec_point_forecast.to_array().transpose("time","lat","lon","variable")
ratio,num_samples = .01, array_forecast.shape[0]
split = int(ratio * num_samples)
print("number training samples:", split * array_forecast.shape[1] * array_forecast.shape[2])
X_tr, y_tr = array_forecast[:split,:,:],array_hindcast[:split,:,:]
X_te, y_te = array_forecast[split:,:,:],array_hindcast[split:,:,:]
print(X_tr.shape,X_te.shape,y_tr.shape,y_te.shape)

#%% New part
model = OceanCurrentGP(current_field)
x,y = units.Distance(m=0.0), units.Distance(m=0.0)
t_0 = datetime.datetime.fromtimestamp(10, datetime.timezone.utc)
t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
ocean_current_vector = OceanCurrentVector(1.0, 1.0)
#%% Train the model:

for i,time in enumerate(y_tr["time"]):
    #Shape elem: lat
    print("training time:",time.item(),i," over ", len(y_tr["time"]))
    for j,lat in enumerate(y_tr["lat"]):
        for k,long in enumerate(y_tr["lon"]):
            model.observe(long.item(),lat.item(),pd.to_datetime(time.item()),OceanCurrentVector(*y_tr[i,j,k].to_numpy()))

print("training is over")
#%% Evaluate the model
diff = []
confidences = []
#all_values = y_te.to_numpy().reshape((-1,2))
locations = np.array(np.meshgrid(y_te["lon"],y_te["lat"],pd.to_datetime(y_te["time"]))).T.reshape((-1,3))
locations[:,2] = pd.to_datetime(locations[:,2])
locations = locations[:100,:]
#model.query_batch(locations)
model.fitting_GP()
model.query_locations(locations)
'''for i,time in enumerate(y_te["time"]):
    if i < 5:
        #Shape elem: lat
        print("evaluation: time:",time.item(),i," over ", len(y_te["time"]))
        for j,lat in enumerate(y_te["lat"]):

            for k,long in enumerate(y_te["lon"]):
                print("INPUTS:", long.item(),lat.item(),pd.to_datetime(time.item()))
                #measurement,confidence = model.query(long.item(), lat.item(), pd.to_datetime(time.item()))
                #hindcast = y_te[i,j,k]
                #diff.append((measurement[0]-hindcast[0],measurement[1]-hindcast[1]))
                #print("conf:",confidence)
                #confidences.append(confidence)'''
print("rmse:", np.array(diff).mean())
#Use MSE to evaluate the diff
#%%

vec_point_forecast = current_field.get_forecast(point=[x.m, y.m], time=t_0)
print('forecast: ', vec_point_forecast)
pre_measurement = model.query(x, y, t_0)
model.observe(x, y, t_0, ocean_current_vector)
post_measurement = model.query(x, y, t_0)
print('results:', pre_measurement, post_measurement)



ocean_field = OceanCurrentField(hindcast_source_dict=source_dict | hindcast_file_config_dict)

ocean_field.hindcast_data_source.grid_dict