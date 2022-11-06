# %%

import os
from datetime import datetime

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

print(os.getcwd())
# %%
radius_in_degree = 0.5
plot_pressure = False

if plot_pressure:
    # NC_file of the pressure
    file = "data_NN/hindcast_pressure_2.nc4"
    field = 'surf_el'
    legend = 'Pressure'
    data = xr.open_dataset(file)
else:
    # NC_file of a  forecast
    file = 'data_ablation_study/fc/august/cmems_mod_glo_phy_anfc_merged-uv_PT1H-i-2022-08-06T12:30:00Z-2022-08-06T12:30:00Z-2022-08-15T23:30:00Z.nc'
    field = 'utotal'
    legend = f"Longitude current - Tile radius {radius_in_degree}°"
    data = xr.open_dataset(file).rename(longitude="lon", latitude="lat")

if 'depth' in data.dims:
    data = data.isel(depth=0)

print(len(data.time))
print(data)


# %%
def plot(data, j=0, add_colorbar=True):
    f, ax = plt.subplots(1)
    print(f"{legend} - data used: {data.isel(time=j).to_array(field)}")
    data.isel(time=j)[[field]].to_array(field).plot(ax=ax, vmin=-1, vmax=1, add_colorbar=add_colorbar)
    # Major ticks every 20, minor ticks every 5
    major_ticks_lon = np.arange(min(data['lon']), max(data['lon']), 2 * radius_in_degree)
    minor_ticks_lon = np.arange(min(data['lon']), max(data['lon']), 1 / 12)
    major_ticks_lat = np.arange(min(data['lat']), max(data['lat']), 2 * radius_in_degree)
    minor_ticks_lat = np.arange(min(data['lat']), max(data['lat']), 1 / 12)

    ax.set_xticks(major_ticks_lon, color='black')
    ax.set_xticks(minor_ticks_lon, minor=True)
    ax.set_yticks(major_ticks_lat, color='black')
    ax.set_yticks(minor_ticks_lat, minor=True)

    # And a corresponding grid
    ax.grid(which='minor', color='silver')
    ax.grid(which='major', color='black')

    # Or if you want different settings for the grids:
    # ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=1)
    print(datetime.fromtimestamp(data.isel(time=j).time.item() // 1_000_000_000))
    ax.set_title(
        f"{legend} - time = {datetime.fromtimestamp(data.isel(time=j).time.item() // 1_000_000_000).strftime('%Y-%m-%d T%H:%M:%S')}")


# %%
idx = 2
plot(data, idx, False)
plot(data, idx + 6, False)
plot(data, idx + 12)
# plot(data, 400)
# %%

print("What is used to print:", data.isel(time=0).to_array(field))

# %%
