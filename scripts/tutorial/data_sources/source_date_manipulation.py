# %% imports
import numpy as np
import xarray as xr

# %% load xarray from .nc file

filepath = "../../../data/nutrients/2021_monthly_nutrients_and_temp.nc"
data_array = xr.open_dataset(filepath)

# %% change year i.e from 2021 to 2022
data_array_ = data_array.assign_coords(time=(data_array.time + np.timedelta64(365, "D")))

# %% save data_array to new .nc file
data_array_.to_netcdf(
    path="../../../data/nutrients/2022_monthly_nutrients_and_temp_faked_from_2021.nc"
)
