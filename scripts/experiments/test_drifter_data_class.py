# %%
from ocean_navigation_simulator.env.data_sources.DrifterData import DrifterData

from datetime import datetime
import random
import folium
from folium import plugins
import xarray as xr

dataset = {
    'host': 'nrt.cmems-du.eu',#ftp host => nrt.cmems-du.eu for Near Real Time products
    'product': 'INSITU_GLO_UV_NRT_OBSERVATIONS_013_048',#name of the In Situ Near Real Time product in the GLO area
    'name': 'drifter',#name of the dataset available in the above In Situ Near Real Time product
    'index_files': ['index_latest.txt', 'index_monthly.txt', 'index_history.txt'],#files describing the content of the lastest, monthly and history netCDF file collections available withint he above dataset
    'index_platform': 'index_platform.txt',#files describing the network of platforms contributing with files in the above collections
}

# define initial area of interest in order to increase speed
targeted_geospatial_lat_min = 17.0  # enter min latitude of your bounding box
targeted_geospatial_lat_max = 32.0  # enter max latitude of your bounding box
targeted_geospatial_lon_min = -75.0 # enter min longitude of your bounding box
targeted_geospatial_lon_max = -100.0 # enter max longitude of your bounding box
targeted_bbox = [targeted_geospatial_lon_min, targeted_geospatial_lat_min, targeted_geospatial_lon_max, targeted_geospatial_lat_max]  # (minx [lon], miny [lat], maxx [lon], maxy [lat])

#define intial time range of interest
targeted_time_range = '2020-01-01T00:00:00Z/2021-01-01T23:59:59Z'

config = {
    'data_dir': '/home/jonas/Documents/Thesis/OceanPlatformControl/data',
    'targeted_bbox': targeted_bbox,
    'area_overlap_type': 'contains',
    'targeted_time_range': targeted_time_range,
    'usr': "mmariuswiggert",
    "pas": "tamku3-qetroR-guwneq"
}
# %%
# download and load index files
drifter_data = DrifterData(dataset, config)
info = drifter_data.index_data
info

# %%
# filter by nc file type ('latest', 'montly', 'history')
targeted_collection = 'monthly'
targeted_collection_info = info[info['file_name'].str.contains(targeted_collection)]
targeted_collection_info.transpose()

# %%
# filter for specific time range
targeted_range = '2020-01-01T00:00:00Z/2021-01-01T23:59:59Z'
targeted_collection_info['timeOverlap'] = targeted_collection_info.apply(drifter_data.temporalOverlap,targeted_time_range=targeted_range,axis=1)
condition = targeted_collection_info['timeOverlap'] == True
subset = targeted_collection_info[condition]
subset.transpose()

# %%
# download selected nc file
nc_link = subset["file_name"].tolist()[0]
print(nc_link)
drifter_data.downloadNCFile(nc_link)

# %%
# read in nc file
path2file = os.path.join(config["data_dir"], "drifter_data", "nc_files", "GL_TS_DC_1301511_202006.nc")
ds = DrifterData.readNCFile(path2file)
ds

# %%
# define area of interest
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
