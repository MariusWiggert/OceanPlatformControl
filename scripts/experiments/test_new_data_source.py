import numpy as np
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import \
    ForecastFileSource, HindcastFileSource, HindcastOpendapSource

# source = ForecastFileSource(source_type='string', config_dict={'folder':"data/cop_fmrc/"})
# source = HindcastFileSource(source_type='string', config_dict={'folder':"data/single_day_hindcasts/"})
# source = HindcastFileSource(source_type='string', config_dict={'folder':"data/hindcast_test/"})

#%%
config_dict = {'source': 'copernicus',
               'currents': 'total', # if we want to take the normal uo, vo currents or 'total' for tide, normal added
               'USERNAME': 'mmariuswiggert', 'PASSWORD': 'tamku3-qetroR-guwneq',
               # 'DATASET_ID': 'global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh',
               'DATASET_ID': 'cmems_mod_glo_phy_anfc_merged-uv_PT1H-i',
               }
source = HindcastOpendapSource(source_type='string', config_dict=config_dict)
#%%
import datetime
t_0 = datetime.datetime(2021, 11, 23, 12, 10, 10, tzinfo=datetime.timezone.utc)
# t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=1)]
x_0 = [-81.5, 23.5, 1]  # lon, lat, battery
x_T = [-80.4, 24.2]
#%%
vec_point = source.get_currents_at_point(point=x_T, time=t_0)
print(vec_point)
#%%
area_xarray = source.get_currents_over_area(x_interval= [-82, -80], y_interval= [24, 26], t_interval=t_interval)
# outside time returns time[0] elements -> can be detected the level above
# outside space will do the same (could do grid check for a warning)
#%% Create the OceanCurrentField Object
from ocean_navigation_simulator.data_sources.OceanCurrentFields import OceanCurrentField

field = OceanCurrentField(source, source)

#%% get data from the field
array = field.get_forecast_area(x_interval= [-82, -80], y_interval= [24, 26], t_interval=t_interval)
#%% get interpolated data
interpol_data = field.get_forecast_area(x_interval= [-82, -80], y_interval= [24, 26], t_interval=t_interval,
                                        spatial_resolution=0.05, temporal_resolution=1000)
