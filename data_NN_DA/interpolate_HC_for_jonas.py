##
import datetime

import numpy as np
import xarray as xr

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

#
arena = ArenaFactory.create(scenario_name="gulf_of_mexico_files_march_august")
fc = arena.ocean_field.forecast_data_source
hc = arena.ocean_field.hindcast_data_source  # .DataArray.interp_like(arena.ocean_field.forecast_data_source.DataArray)
j = 30
lat_interv = fc.DataArray['lat'][[j, -j]]
lon_interv = fc.DataArray['lon'][[j, -j]]
margin_space = 1 / 12
lat_interv_hc = (fc.DataArray['lat'][[j, -j]][0] - margin_space, fc.DataArray['lat'][[j, -j]][1] + margin_space)
lon_interv_hc = (fc.DataArray['lon'][[j, -j]][0] - margin_space, fc.DataArray['lon'][[j, -j]][1] + margin_space)

start = datetime.datetime(2022, 4, 2, 12, 30, tzinfo=datetime.timezone.utc) + datetime.timedelta(hours=1)
delta = datetime.timedelta(1)
margin = datetime.timedelta(hours=1)
res = xr.Dataset()
len_days = 120
for i in range(len_days):
    if not i % 10:
        print(i + 1, "/", len_days)
    try:
        f = fc.get_data_over_area(lon_interv, lat_interv, [start, start + delta]).isel(time=slice(0, 24))

        lat = f['lat']
        lon = f['lon']
        lat_2 = np.arange(20, 28 + 1 / 12, 1 / 12)
        lon_2 = np.arange(-96, -76 + 1 / 12, 1 / 12)
        # lat_2 = np.concatenate((np.arange(20, lat[0], 1 / 12), lat, np.arange(1 / 12 + lat[-1], 28, 1 / 12)))
        # lon_2 = np.concatenate((np.arange(-97, lon[0], 1 / 12), lon, np.arange(1 / 12 + lon[-1], -77, 1 / 12)))
        time_2 = f["time"]

        base = xr.Dataset(
            data_vars=dict(
                water_u=(("lon", "lat", "time"), np.full([len(lon_2), len(lat_2), len(time_2)], np.nan)),
                water_v=(("lon", "for ", "time"), np.full([len(lon_2), len(lat_2), len(time_2)], np.nan)),
            ),
            coords=dict(
                lon=lon_2,
                lat=lat_2,
                time=time_2,
            ))

        f = f.interp_like(base)
        h = hc.get_data_over_area(lon_interv_hc, lat_interv_hc, [start - margin, start + delta + margin])
        h_interp = h.interp_like(f)
        error = f - h_interp
        res = error.combine_first(res)
    except ValueError:
        print(f"error with day: {start}. Skipped!")
    start += delta

res.to_netcdf('forecast_c3_script/export_error.nc')
tile_radius = 4
all_arrays = []
for lo in range(-96, -76, tile_radius):
    for la in range(20, 28, tile_radius):
        for i in range(0, len(res["time"]), 24):
            all_arrays.append(
                res.sel(lon=slice(lo, lo + tile_radius), lat=slice(la, la + tile_radius)).isel(time=slice(i, i + 24)))

print(all_arrays)
path = "forecast_c3_script/multiples_sub_arrays/"
for i, ar in enumerate(all_arrays):
    ar.to_netcdf(path + f"export_error_{i}.nc")

print("over :)")
