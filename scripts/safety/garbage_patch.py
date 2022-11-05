import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


# TODO: replace with the function form datasource
# @staticmethod
def set_up_geographic_ax() -> plt.axes:
    """Helper function to set up a geographic ax object to plot on."""

    ax = plt.axes(projection=ccrs.PlateCarree())
    grid_lines = ax.gridlines(draw_labels=True, zorder=5)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    ax.add_feature(cfeature.LAND, zorder=3, edgecolor="black")
    # TODO: remove hardcoded pos
    ax.set_extent([-175, -100, 15, 45], ccrs.PlateCarree())
    return ax


# TODO: see if this is indeed best dataset
df = pd.read_csv("data/lebreton/Lebreton2018_HistoricalDataset.csv")
df["lat"] = df["Latitude (degrees)"]
df["lon"] = df["Longitude (degrees)"]
df = df[df["Inside GPGP?"] == 1]
ds = xr.Dataset.from_dataframe(df)

# Plotting with cartopy
central_lat = 30
central_lon = -145
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title("Australia")
ax.set_extent([-175, -100, 15, 45], ccrs.PlateCarree())
ax.coastlines(resolution="110m")
ax.scatter(ds["lon"], ds["lat"])
# plt.plot(central_lon, central_lat, markersize=2, marker="o", color="red")
plt.show()
