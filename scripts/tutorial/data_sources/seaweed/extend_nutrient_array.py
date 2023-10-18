#%% Run imports
import datetime
import numpy as np
import xarray as xr
import glob

#%% load xarray of the average data from the filepath 'ocean_navigation_simulator/package_data/monthly_averages/'
nc_files = glob.glob('./ocean_navigation_simulator/package_data/nutrients/'  + "*.nc")
nc_files = sorted(nc_files, key=lambda x: xr.open_dataset(x).time[0].values)

#%% # get the averages for new years time for all variables
GlobalArray = xr.open_mfdataset(nc_files)
new_year_no3 = (GlobalArray['no3'].data[-1] + GlobalArray['no3'].data[0])/2
new_year_po4 = (GlobalArray['po4'].data[-1] + GlobalArray['po4'].data[0])/2
new_year_temp = (GlobalArray['Temperature'].data[-1] + GlobalArray['Temperature'].data[0])/2

#%% Now we need to add the new year to the beginning of the array
DataArray = xr.open_dataset(nc_files[0])
# create a new dataset that is longer by the length of the first and last element
new_no3 = np.concatenate((new_year_no3.reshape(1,1,681,1440), DataArray['no3'].data), axis=0)
new_po4 = np.concatenate((new_year_po4.reshape(1,1,681,1440), DataArray['po4'].data), axis=0)
new_temp = np.concatenate((new_year_temp.reshape(1,681,1440), DataArray['Temperature'].data), axis=0)
#%% create the new time axis
# transform from datetime object to np.datetime64
np_datetime = np.datetime64(datetime.datetime(2022, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)).reshape(1,)
new_time_axis = np.concatenate((np_datetime, DataArray.coords['time'].data), axis=0)
#%%
new_ds = xr.Dataset(
    data_vars=dict(
        no3=(['time', 'depth', 'latitude', 'longitude'], new_no3, DataArray['no3'].attrs),
        # now the same for po4 and Temperature
        po4=(['time', 'depth', 'latitude', 'longitude'], new_po4, DataArray['po4'].attrs),
        Temperature=(['time', 'latitude', 'longitude'], new_temp, DataArray['Temperature'].attrs),
    ),
    coords=dict(
        depth= DataArray.coords['depth'],
        longitude=DataArray.coords['longitude'],
        latitude=DataArray.coords['latitude'],
        time=new_time_axis
    ),
    attrs=DataArray.attrs,
)

# now save it to netcdf
name = '2022_monthly_nutrients_and_temp_155m_faked_from_2021_jan_jun_new.nc'
new_ds.to_netcdf('./ocean_navigation_simulator/package_data/nutrients/' + name)

#%% do it for the last element
DataArray = xr.open_dataset(nc_files[1])
#%% Now we need to add the new year to the end of the array
new_no3 = np.concatenate((DataArray['no3'].data, new_year_no3.reshape(1,1,681,1440)), axis=0)
new_po4 = np.concatenate((DataArray['po4'].data, new_year_po4.reshape(1,1,681,1440)), axis=0)
new_temp = np.concatenate((DataArray['Temperature'].data, new_year_temp.reshape(1,681,1440)), axis=0)
# Now create a new xarray dataset with the new data
#%%
from ocean_navigation_simulator.utils.units import get_datetime_from_np64
# transform from datetime object to np.datetime64
np_datetime = np.datetime64(datetime.datetime(2022, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)).reshape(1,)
new_time_axis = np.concatenate((DataArray.coords['time'].data, np_datetime), axis=0)

#%%
new_ds = xr.Dataset(
    data_vars=dict(
        no3=(['time', 'depth', 'latitude', 'longitude'], new_no3, DataArray['no3'].attrs),
        # now the same for po4 and Temperature
        po4=(['time', 'depth', 'latitude', 'longitude'], new_po4, DataArray['po4'].attrs),
        Temperature=(['time', 'latitude', 'longitude'], new_temp, DataArray['Temperature'].attrs),
    ),
    coords=dict(
        depth= DataArray.coords['depth'],
        longitude=DataArray.coords['longitude'],
        latitude=DataArray.coords['latitude'],
        time=new_time_axis
    ),
    attrs=DataArray.attrs,
)

# now save it to netcdf
name = '2022_monthly_nutrients_and_temp_155m_faked_from_2021_jul_dec_new.nc'
new_ds.to_netcdf('./ocean_navigation_simulator/package_data/nutrients/' + name)

#%% testing if it works
#%%
nc_files = glob.glob('./ocean_navigation_simulator/package_data/nutrients/'  + "*.nc")
nc_files = sorted(nc_files, key=lambda x: xr.open_dataset(x).time[0].values)

GlobalArray = xr.open_mfdataset(nc_files)

array = (GlobalArray['no3'].data[0] == GlobalArray['no3'].data[-1]).compute()
import matplotlib.pyplot as plt
plt.imshow(array[0,...])
plt.show()