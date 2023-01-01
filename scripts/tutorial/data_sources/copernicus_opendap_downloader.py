import datetime
import os
from typing import List

import numpy as np
import xarray as xr

from ocean_navigation_simulator.data_sources.OceanCurrentField import (
    OceanCurrentField,
)

hindcast_dict = {
    "field": "OceanCurrents",
    "source": "opendap",
    "source_settings": {
        "service": "copernicus",
        "currents": "total",
        "USERNAME": "mmariuswiggert",
        "PASSWORD": "tamku3-qetroR-guwneq",
        "DATASET_ID": "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
    },
}


# casadi_cache_dict: containing the cache settings to use in the sources for caching of 3D data
# e.g. {'deg_around_x_t': 2, 'time_around_x_t': 3600*24*12}"""

casadi_cache_dict = {"deg_around_x_t": 2, "time_around_x_t": 3600 * 24 * 1}


def get_copernicus_hindcast(
    lon_range: List[float],
    lat_range: List[float],
    start: datetime.datetime,
    days: int,
    save_dir: str,
):
    """Gets 1 day long copernicus hindcasts for each day specified. E.g. if days=2, it saves two files of length 1
    days, the second one starting one day after the first.
    Note 1: Uses Marius' HindcastOpendapSource class which changes the internal xarray naming, this func changes it
    back by creating a new xarray and saving it -> consider changing this!
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for day in range(days):
        file_start = start + datetime.timedelta(days=day)
        file_end = file_start + datetime.timedelta(days=1)
        print(f"Downloading for time period of [{file_start}, {file_end}].")
        file_name = (
            f"copernicus_hindcast_lon_{lon_range}_lat_{lat_range}_time_"
            + f"[{start + datetime.timedelta(days=day)},{start + datetime.timedelta(days=1 + day)}].nc"
        )
        # check if file already exists
        if os.path.exists(os.path.join(save_dir, file_name)):
            print("File already exists!")
            continue

        # need to run OceanCurrentField again so connection does not time out -> makes loop very slow
        ocean_field = OceanCurrentField(
            hindcast_source_dict=hindcast_dict, casadi_cache_dict=casadi_cache_dict
        )
        print("Made connection to Copernicus!")
        # get xarray
        ds_hindcast = ocean_field.hindcast_data_source.DataArray
        # slice for specific range
        hindcast = ds_hindcast.sel(
            time=slice(
                start + datetime.timedelta(days=day), start + datetime.timedelta(days=1 + day)
            ),
            lon=slice(*lon_range),
            lat=slice(*lat_range),
        )

        water_u = hindcast["water_u"].values[:, np.newaxis, :, :]
        water_v = hindcast["water_v"].values[:, np.newaxis, :, :]
        attrs = hindcast.attrs

        # Need to create new xarray to change data variable names back to be the same as copernicus forecast
        hindcast = xr.Dataset(
            data_vars=dict(
                utotal=(["time", "depth", "latitude", "longitude"], water_u),
                vtotal=(["time", "depth", "latitude", "longitude"], water_v),
            ),
            coords=dict(
                time=("time", hindcast.coords["time"].values),
                depth=("depth", [hindcast.coords["depth"].values]),
                latitude=("latitude", hindcast.coords["lat"].values),
                longitude=("longitude", hindcast.coords["lon"].values),
            ),
            attrs=attrs,
        )
        # save data to file
        hindcast.to_netcdf(os.path.join(save_dir, file_name), engine="netcdf4")
        print(f"Written: {file_name}.")
    print("Finished downloads.")


if __name__ == "__main__":
    save_dir = "path_name"
    lon_range = [-120, -70]
    lat_range = [-20, 20]
    start = datetime.datetime(2022, 11, 30, 00, 0, 0)
    days = 20
    get_copernicus_hindcast(lon_range, lat_range, start, days, save_dir=save_dir)
