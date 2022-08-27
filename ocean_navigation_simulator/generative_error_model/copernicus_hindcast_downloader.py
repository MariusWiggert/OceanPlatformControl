from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.generative_error_model.utils import load_config, get_path_to_project

import datetime
import os
import xarray as xr
import numpy as np

relative_path = "scenarios/generative_error_model/config_buoy_data.yaml"
config_path = os.path.join(get_path_to_project(os.getcwd()), relative_path)
config = load_config(config_path)

hindcast_dict = config["copernicus_opendap"]["hindcast"]
sim_cache_dict = config["sim_cache_dict"]

root_dir = os.path.join(get_path_to_project(os.getcwd()), "data/drifter_data/hindcasts/area1/")
days = 1
start = datetime.datetime(2022, 4, 21, 12, 30, 0)
lon_range = [-145, -115]
lat_range = [15, 35]

for day in range(days):
    print(f"Downloading for [{start}, {start+datetime.timedelta(days=9)}].")

    # need to run OceanCurrentField again so connection does not time out
    ocean_field = OceanCurrentField(hindcast_source_dict=hindcast_dict, sim_cache_dict=sim_cache_dict)
    # get xarray
    ds_hindcast = ocean_field.hindcast_data_source.DataArray
    # slice for specific range
    hindcast = ds_hindcast.sel(time=slice(start+datetime.timedelta(days=day), start+datetime.timedelta(days=9+day)),
                               lon=slice(*lon_range),
                               lat=slice(*lat_range))

    water_u = hindcast["water_u"].values[:, np.newaxis, :, :]
    water_v = hindcast["water_v"].values[:, np.newaxis, :, :]
    attrs = hindcast.attrs
    attrs["source"] = "HYCOM"

    hindcast = xr.Dataset(
        data_vars=dict(
            water_u=(["time", "depth", "lat", "lon"], water_u),
            water_v=(["time", "depth", "lat", "lon"], water_v)
        ),
        coords=dict(
            time=("time", hindcast.coords["time"].values),
            depth=("depth", [hindcast.coords["depth"].values]),
            lat=("lat", hindcast.coords["lat"].values),
            lon=("lon", hindcast.coords["lon"].values)
        ),
        attrs=attrs
    )
    file_name = f"copernicus_hindcast_lon_{lon_range}_lat_{lat_range}_time_[{start+datetime.timedelta(days=day)},{start+datetime.timedelta(days=9+day)}].nc"
    # save data to file
    hindcast.to_netcdf(os.path.join(root_dir, file_name), engine="netcdf4")
    print(f"Written: {file_name}.")
print("Finished downloads.")
