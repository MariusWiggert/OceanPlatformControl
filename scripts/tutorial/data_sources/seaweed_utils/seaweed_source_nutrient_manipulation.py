# %% imports
import numpy as np
import xarray as xr

""""This script creates a toy nutrient file which can be then used for easier debugging"""

# %% load xarray from .nc file

filepath = "../../../data/nutrients/2022_monthly_nutrients_and_temp_faked_from_2021.nc"
data_array = xr.open_dataset(filepath)

print(data_array["no3"].loc[dict(longitude=-138.5, latitude=39.5)])
print(data_array["no3"].loc[dict(longitude=-139, latitude=35)])
print(data_array["no3"].loc[dict(longitude=-1, latitude=3)])
print(data_array["no3"].loc[dict(longitude=-1, latitude=-1)])


# %% get relevant data
data_optimal_no3 = data_array["no3"].loc[dict(longitude=-138.5, latitude=39.5)]
data_suboptimal_no3 = data_array["no3"].loc[dict(longitude=-139, latitude=35)]

print(data_optimal_no3)
print(data_suboptimal_no3)

data_optimal_po4 = data_array["po4"].loc[dict(longitude=-138.5, latitude=39.5)]
data_suboptimal_po4 = data_array["po4"].loc[dict(longitude=-139, latitude=35)]

print(data_optimal_po4)
print(data_suboptimal_po4)

data_optimal_temp = data_array["Temperature"].loc[dict(longitude=-138.5, latitude=39.5)]
data_suboptimal_temp = data_array["Temperature"].loc[dict(longitude=-139, latitude=35)]

print(data_optimal_temp)
print(data_suboptimal_temp)


# %% Manipulate data

# modify a 2D region using xr.where()
mask = (
    (data_array.coords["latitude"] >= -90)
    & (data_array.coords["latitude"] <= 90)
    & (data_array.coords["longitude"] >= -180)
    & (data_array.coords["longitude"] <= 180)
)


data_array["no3"] = xr.where(mask, data_suboptimal_no3, data_array["no3"])
data_array["po4"] = xr.where(mask, data_suboptimal_po4, data_array["po4"])
data_array["Temperature"] = xr.where(mask, data_suboptimal_temp, data_array["Temperature"])


# overwrite certain region with benefical nutrient values

mask = (
    (data_array.coords["latitude"] >= 1)
    & (data_array.coords["latitude"] <= 4)
    & (data_array.coords["longitude"] >= -2)
    & (data_array.coords["longitude"] <= 1)
)


data_array["no3"] = xr.where(mask, data_optimal_no3, data_array["no3"])
data_array["po4"] = xr.where(mask, data_optimal_po4, data_array["po4"])
data_array["Temperature"] = xr.where(mask, data_optimal_temp, data_array["Temperature"])

print(data_array["no3"].loc[dict(longitude=-138.5, latitude=39.5)])
print(data_array["no3"].loc[dict(longitude=-139, latitude=35)])
print(data_array["no3"].loc[dict(longitude=-1, latitude=3)])
print(data_array["no3"].loc[dict(longitude=-1, latitude=-1)])


# %% save data_array to new .nc file
data_array.to_netcdf(
    path="../../../data/nutrients/2022_monthly_nutrients_and_temp_faked_from_2021_certain_region_adpated_for_testing.nc"
)


# %%
