# %%

import os

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

print(os.getcwd())
# %%
file = "data_NN/hindcast_pressure.nc4"
data = xr.open_dataset(file)


# %%
def plot(data, j=0):
    f, ax = plt.subplots(1)
    data.isel(time=j).to_array("surf_el").plot(ax=ax)
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


# %%
plot(data, 10)
plot(data, 16)
plot(data, 22)
# plot(data, 400)
# %%
