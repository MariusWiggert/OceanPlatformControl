# %%

import os
from datetime import datetime

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

print(os.getcwd())
# %%
file = "data_NN/hindcast_pressure_2.nc4"
data = xr.open_dataset(file)
print(len(data.time))


# %%
def plot(data, j=0, add_colorbar=True):
    f, ax = plt.subplots(1)
    print(f"data used: {data.isel(time=j).to_array('surf_el')}")
    data.isel(time=j).to_array("surf_el").plot(ax=ax, vmin=-1, vmax=1, add_colorbar=add_colorbar)
    # Major ticks every 20, minor ticks every 5
    major_ticks_lon = np.arange(min(data['lon']), max(data['lon']), 24 * 1 / 12)
    minor_ticks_lon = np.arange(min(data['lon']), max(data['lon']), 1 / 12)
    major_ticks_lat = np.arange(min(data['lat']), max(data['lat']), 24 * 1 / 12)
    minor_ticks_lat = np.arange(min(data['lat']), max(data['lat']), 1 / 12)

    ax.set_xticks(major_ticks_lon)
    ax.set_xticks(minor_ticks_lon, minor=True)
    ax.set_yticks(major_ticks_lat)
    ax.set_yticks(minor_ticks_lat, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=1)
    print(datetime.fromtimestamp(data.isel(time=j).time.item() // 1_000_000_000))
    ax.set_title(
        f"time = {datetime.fromtimestamp(data.isel(time=j).time.item() // 1_000_000_000).strftime('%Y-%m-%d T%H:%M:%S')}")


# %%
idx = 2
plot(data, idx, False)
plot(data, idx + 6, False)
plot(data, idx + 12)
# plot(data, 400)
# %%

print("What is used to print:", data.isel(time=0).to_array('surf_el'))

# %%
