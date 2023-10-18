# %% imports
import xarray as xr

# %% load xarray from .nc file

filepath = "data/nutrients/2022_monthly_nutrients_and_temp_155m_faked_from_2021.nc"
data_array = xr.open_dataset(filepath)

# %% calculate midpoint of time dimension and create two datasets with each half the data
midpoint = data_array.time.size // 2


ds1 = data_array.isel(time=slice(None, midpoint))
ds2 = data_array.isel(time=slice(midpoint, None))
# %% save data_array to new .nc file
ds1.to_netcdf(
    path="ocean_navigation_simulator/package_data/nutrients/2022_monthly_nutrients_and_temp_155m_faked_from_2021_jan_jun.nc"
)
ds2.to_netcdf(
    path="ocean_navigation_simulator/package_data/nutrients/2022_monthly_nutrients_and_temp_155m_faked_from_2021_jul_dec.nc"
)

# %%
