from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.generative_error_model.utils import load_config, get_path_to_project

import datetime
import os
import xarray as xr
import numpy as np
from typing import List


def get_copernicus_hindcast(lon_range: List[float], lat_range: List[float], start: datetime.datetime,
                            days:int, save_dir: str):

    relative_path = "scenarios/generative_error_model/config_buoy_data.yaml"
    config_path = os.path.join(get_path_to_project(os.getcwd()), relative_path)
    config = load_config(config_path)

    hindcast_dict = config["copernicus_opendap"]["hindcast"]
    sim_cache_dict = config["sim_cache_dict"]

    root_dir = os.path.join(get_path_to_project(os.getcwd()), save_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for day in range(days):
        file_start = start + datetime.timedelta(days=day)
        file_end = file_start + datetime.timedelta(days=9)
        print(f"Downloading for time period of [{file_start}, {file_end}].")
        file_name = f"copernicus_hindcast_lon_{lon_range}_lat_{lat_range}_time_[{start + datetime.timedelta(days=day)},{start + datetime.timedelta(days=9 + day)}].nc"
        if os.path.exists(os.path.join(root_dir, file_name)):
            print(f"File already exists!")
            continue

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
        # set to HYCOM since variables already have HYCOM naming otherwise OceanCurrentSource will fail.
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
        # save data to file
        hindcast.to_netcdf(os.path.join(root_dir, file_name), engine="netcdf4")
        print(f"Written: {file_name}.")
    print("Finished downloads.")


if __name__ == "__main__":
    save_dir = "data/drifter_data/hindcasts/area3/"
    # lon_range = [-150, -115]
    # lat_range = [0, 40]
    lon_range = [-120, -90]
    lat_range = [-15, 15]
    start = datetime.datetime(2022, 5, 2, 12, 30, 0)
    days = 166
    get_copernicus_hindcast(lon_range, lat_range, start, days, save_dir=save_dir)
