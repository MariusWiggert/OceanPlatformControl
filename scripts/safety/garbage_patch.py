import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import ConvexHull


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


def plot_convex_hull(points, ax):
    """Plot a convex hull"""
    hull = ConvexHull(points)
    plt.plot(points[:, 0], points[:, 1], "o")
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], "k-")
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], "r--", lw=2)
    plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], "ro")
    plt.show()


if __name__ == "__main__":
    # TODO: see if this is indeed best dataset
    df = pd.read_csv("data/lebreton/Lebreton2018_HistoricalDataset.csv")
    df["lat"] = df["Latitude (degrees)"]
    df["lon"] = df["Longitude (degrees)"]
    df = df[df["Inside GPGP?"] == 1]
    ds = xr.Dataset.from_dataframe(df)

    # Plot a convex hull around garbage patch
    ax = set_up_geographic_ax()
    points = np.column_stack((ds["lon"], ds["lat"]))
    plot_convex_hull(points, ax)

    # # Plotting with cartopy
    # central_lat = 30
    # central_lon = -145
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_extent([-175, -100, 15, 45], ccrs.PlateCarree())
    # ax.coastlines(resolution="110m")
    # ax.scatter(ds["lon"], ds["lat"])
    # plt.show()
