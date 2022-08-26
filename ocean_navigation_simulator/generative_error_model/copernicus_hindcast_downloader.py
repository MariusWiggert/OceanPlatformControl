from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.generative_error_model.utils import load_config, get_path_to_project

import datetime
import os

relative_path = "scenarios/generative_error_model/config_buoy_data.yaml"
config_path = os.path.join(get_path_to_project(os.getcwd()), relative_path)
config = load_config(config_path)

hindcast_dict = config["copernicus_opendap"]["hindcast"]
sim_cache_dict = config["sim_cache_dict"]

root_dir = "/home/jonas/Downloads/temp"
days = 30
start = datetime.datetime(2022, 4, 21, 12, 30, 0)
lon_range = [-140, -120]
lat_range = [20, 30]

for day in range(days):
    print(f"Downloading for [{start}, {start+datetime.timedelta(days=9)}].")

    # need to run OceanCurrentField again so connection does not time out
    ocean_field = OceanCurrentField(hindcast_source_dict=hindcast_dict, sim_cache_dict=sim_cache_dict)
    # get xarray
    ds_hindcast = ocean_field.hindcast_data_source.DataArray
    # slice for specific range
    hindcast = ds_hindcast.sel(time=slice(start+datetime.timedelta(days=day), start+datetime.timedelta(days=9+day)),
                               lon=slice(*lon_range),
                               lat=slice(*lat_range)).drop_vars('depth')
    file_name = f"copernicus_hindcast_lon_{lon_range}_lat_{lat_range}_time_[{start+datetime.timedelta(days=day)},{start+datetime.timedelta(days=9+day)}].nc"
    # save data to file
    hindcast.to_netcdf(os.path.join(root_dir, file_name))
    start += datetime.timedelta(days=1)

print("Finished downloads.")
