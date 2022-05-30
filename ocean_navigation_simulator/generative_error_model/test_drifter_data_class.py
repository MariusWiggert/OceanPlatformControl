'''
Script to test the drifter data class.

By changing the "dataset" value in the "config" dictionary the data source can be
switched from "dataset_difter" data (Global Drifter Program) to "dataset_argo" data (Global Data Assembly Center).

The config dictionary contains settings for the index files such as the initial
area of interest and time range of interest. One can also specify the data_dir.
'''

# %%
# imports
import random
from datetime import datetime

import folium
import xarray as xr
from folium import plugins
from ocean_navigation_simulator.env.data_sources.DrifterData import DrifterData
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

dataset_drifter = {
    'host': 'nrt.cmems-du.eu',#ftp host => nrt.cmems-du.eu for Near Real Time products
    'product': 'INSITU_GLO_UV_NRT_OBSERVATIONS_013_048',#name of the In Situ Near Real Time product in the GLO area
    'name': 'drifter',# name of the dataset available in the above In Situ Near Real Time product
    'index_files': ['index_latest.txt', 'index_monthly.txt', 'index_history.txt'], #files describing the content of the lastest, monthly and history netCDF file collections available withint he above dataset
    'index_platform': 'index_platform.txt', #files describing the network of platforms contributing with files in the above collections
}
dataset_argo = {
    'host': 'nrt.cmems-du.eu',#ftp host => nrt.cmems-du.eu for Near Real Time products
    'product': 'INSITU_GLO_UV_NRT_OBSERVATIONS_013_048',#name of the In Situ Near Real Time product in the GLO area
    'name': 'argo',# name of the dataset available in the above In Situ Near Real Time product
    'index_files': ['index_history.txt'], #files describing the content of the lastest, monthly and history netCDF file collections available withint he above dataset
}

# define initial area of interest in order to increase speed
targeted_geospatial_lat_min = 17.0  # enter min latitude of your bounding box
targeted_geospatial_lat_max = 32.0  # enter max latitude of your bounding box
targeted_geospatial_lon_min = -75.0 # enter min longitude of your bounding box
targeted_geospatial_lon_max = -100.0 # enter max longitude of your bounding box
targeted_bbox = [targeted_geospatial_lon_min, targeted_geospatial_lat_min, targeted_geospatial_lon_max, targeted_geospatial_lat_max]  # (minx [lon], miny [lat], maxx [lon], maxy [lat])

# define intial time range of interest
targeted_time_range = '2021-01-01T00:00:00Z/2021-01-31T23:59:59Z'

config = {
    'data_dir': '/home/jonas/Documents/Thesis/OceanPlatformControl/data',
    'dataset': dataset_drifter,
    'targeted_bbox': targeted_bbox,
    'area_overlap_type': 'intersects',
    'targeted_time_range': targeted_time_range,
    'usr': "mmariuswiggert",
    "pas": "tamku3-qetroR-guwneq"
}
# %%
# download and load index files
drifter_data = DrifterData(config)
info = drifter_data.index_data
info.transpose()

# %%
# filter by nc file type ('latest', 'montly', 'history')
targeted_collection = 'monthly'
targeted_collection_info = info[info['file_name'].str.contains(targeted_collection)]
targeted_collection_info.transpose()

# %%
# download selected nc file
nc_link = targeted_collection_info["file_name"].tolist()[0]
file_name = drifter_data.downloadNCFile(nc_link)

# %%
# read in nc file
path2file = os.path.join(config["data_dir"], "drifter_data", "nc_files", file_name)
ds = DrifterData.readNCFile(path2file)
ds

# %%
# plotting map of drifters

fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection=ccrs.PlateCarree())
grid_lines = ax.gridlines(draw_labels=True, zorder=5)
grid_lines.top_labels = False
grid_lines.right_labels = False
ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black')

u = ds["LONGITUDE"]
v = ds["LATITUDE"]
ax.plot(u,v)
plt.show()

# %%
# plot velocities of radar data
velocity = (ds["NSCT"]**2 + ds["EWCT"]**2)**0.5
velocity.isel(TIME=5).plot()
display(velocity.sel(LATITUDE=45.0, method='nearest').mean().values)
# can also use sel() method with kwarg method='nearest'

# %%
# plot individual directions of radar data
u = ds["EWCT"]
v = ds["NSCT"]

fig = plt.figure(figsize=(14,6))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

u.isel(TIME=5).sel(LATITUDE=slice(40,42), LONGITUDE=slice(-71,-65)).plot(ax=ax1)
v.isel(TIME=5).sel(LATITUDE=slice(40,42), LONGITUDE=slice(-71,-65)).plot(ax=ax2)

plt.show()
