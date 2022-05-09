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
targeted_time_range = '2010-01-01T00:00:00Z/2021-01-01T23:59:59Z'

config = {
    'data_dir': '/home/jonas/Documents/Thesis/OceanPlatformControl/data',
    'dataset': dataset_drifter,
    'targeted_bbox': targeted_bbox,
    'area_overlap_type': 'contains',
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
targeted_collection = 'history'
targeted_collection_info = info[info['file_name'].str.contains(targeted_collection)]
targeted_collection_info.transpose()

# %%
# download selected nc file
nc_link = targeted_collection_info["file_name"].tolist()[3]
file_name = drifter_data.downloadNCFile(nc_link)

# %%
# read in nc file
path2file = os.path.join(config["data_dir"], "drifter_data", "nc_files", file_name)
ds = DrifterData.readNCFile(path2file)
ds

# %%
# play with data in NC file
for var in ds.variables:
    print(f"{var}: {ds[var].attrs['long_name']}")

start = '2020-06-29'
end = '2020-06-30'
subset = ds['NSCT'] #.sel(TIME=slice(start,end))

# check where depth is 15.0 metres
subset_depth = ds['DEPH']
subset_depth

# filter for correct depth
subset = subset.where(subset_depth == 15.0)
subset

# %%
# check quality flags of parameter (NSCT) -> one means "good data"
subset_QC = ds['NSCT_QC'][:,2]
subset_QC
subset_QC.plot()

# %%
# check if data is good for position
subset_ps_QC = ds['POSITION_QC'].plot()

# %%
lats = subset['LATITUDE'].where(subset['POSITION_QC'] == 1).values.tolist()
lats = [i[0] for i in lats]
lons = subset['LONGITUDE'].where(subset['POSITION_QC'] == 1).values.tolist()
lons = [i[1] for i in lons]
times = subset['TIME'].where(subset['TIME_QC'] == 1).values.tolist()
strtimes = subset['TIME'].where(subset['TIME_QC'] == 1).values[:]

# %%
# aggregate data in NetCFD


# %%
# define area of interest
subset = info
upper_left = [targeted_geospatial_lat_max, targeted_geospatial_lon_min]
upper_right = [targeted_geospatial_lat_max, targeted_geospatial_lon_max]
lower_right = [targeted_geospatial_lat_min, targeted_geospatial_lon_max]
lower_left = [targeted_geospatial_lat_min, targeted_geospatial_lon_min]
edges_ = [upper_left, upper_right, lower_right, lower_left]

# plot if the bbox of files overlaps with targeted area
m = folium.Map(location=[39.0, 0], zoom_start=6)
m.add_child(folium.vector_layers.Polygon(locations=edges_))
for platform, files in subset.groupby(['platform_code', 'data_type']):
    color = "%06x" % random.randint(0, 0xFFFFFF)
    for i in range(0, len(files)):
        netcdf = files.iloc[i]['file_name'].split('/')[-1]
        upper_left = [
            files.iloc[i]['geospatial_lat_max'],
            files.iloc[i]['geospatial_lon_min']
        ]
        upper_right = [
            files.iloc[i]['geospatial_lat_max'],
            files.iloc[i]['geospatial_lon_max']
        ]
        lower_right = [
            files.iloc[i]['geospatial_lat_min'],
            files.iloc[i]['geospatial_lon_max']
        ]
        lower_left = [
            files.iloc[i]['geospatial_lat_min'],
            files.iloc[i]['geospatial_lon_min']
        ]
        edges = [upper_left, upper_right, lower_right, lower_left]
        popup_info = '<b>netcdf</b>: ' + files.iloc[i]['netcdf']
        m.add_child(folium.vector_layers.Polygon(locations=edges,color='#' + color,popup=(folium.Popup(popup_info))))
        m.fit_bounds(edges, max_zoom=6)
DrifterData.plotFoliumMapObject(m)


# %%
# plotting map of drifters
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# targeted_bbox (minx [lon], miny [lat], maxx [lon], maxy [lat])

fig = plt.figure(figsize=(6,6))

# ax = plt.axes(projection=ccrs.PlateCarree())
# grid_lines = ax.gridlines(draw_labels=True, zorder=5)
# grid_lines.top_labels = False
# grid_lines.right_labels = False
# ax.add_feature(cfeature.LAND, zorder=3, edgecolor='black')


# checking if data is good (TODO: find way to get correct depth)
da_nsct = ds["NSCT"][:,1] #.where(ds['POSITION_QC'] == 1)

da_nsct = da_nsct.expand_dims({"LATITUDE": (ds['LATITUDE']), "LONGITUDE": (ds["LONGITUDE"])}, axis=[1,2])

da_nsct[0].plot()
da_nsct

# plt.show()
# %%
