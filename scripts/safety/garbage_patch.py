from typing import AnyStr, List, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import ConvexHull, Delaunay


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


def alpha_complex(points: np.array, alpha: int, only_outer: bool = True) -> set():
    """Generate alpha compled around set of points.
    Reference:https://youtu.be/-XCVn73p3xs?t=633

    Args:
        points (np.array): (n,2) points
        alpha (int): Alpha value.
        only_outer (bool, optional): Keep only outer edge or also interior edges. Defaults to True.

    Returns:
        set(): Pairs of indexes of the points (i,j) that represent edges of alpha shape.
    """
    assert points.shape[0] >= 4, "4 or more points required"

    def add_edge(edges: set(), i: int, j: int) -> None:
        """
        Add edge between points i and j if edge not already in set.
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Each edge only once"
            if only_outer:
                # If both neighboring triangles are within shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    # Perform tesselation of space with delauny complex
    tri = Delaunay(points)
    edges = set()
    # Loop over triangles of delauny complex
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Compute side lengths of triangle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Area of triangle (herons formula)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # The three points of the dealuny complex triangle define a circumcircle
        # We keep all triangles that have a smaller radius than the alpha value
        r = a * b * c / (4.0 * area)
        if r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def plot_convex_hull(points, ax):
    """Plot a convex hull"""
    hull = ConvexHull(points)
    plt.plot(points[:, 0], points[:, 1], "o")
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], "k-")
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], "r--", lw=2)
    plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], "ro")
    plt.show()


def plot_alpha_complex(points, ax, alpha):
    edges = alpha_complex(points, alpha=alpha)
    plt.plot(points[:, 0], points[:, 1], ".")
    for i, j in edges:
        plt.plot(points[[i, j], 0], points[[i, j], 1])
    plt.text(270.5, 459, "alpha=1", size=18)
    plt.show()


def create_xarray_from_csv(
    filename: str = "data/lebreton/Lebreton2018_HistoricalDataset.csv",
) -> xr:
    df = pd.read_csv(filename)
    df = df.rename(columns={"Latitude (degrees)": "lat", "Longitude (degrees)": "lon"})
    df = df[df["Inside GPGP?"] == 1]
    ds = xr.Dataset.from_dataframe(df)
    return ds


# Either take range of vertices + padding or take user defined ranges for where to plot
# TODO: only GPGP region or global (both with only GPGP data)?
# TODO: figure out why alpha complex with alpha value zero does not return the complex hull
# TODO: there will be errors in areas where the convex hull should wrap around, hence at +-180°


def create_xarray_mask(
    xarray: xr,
    x_range: List[float],
    y_range: List[float],
    res_lat: float = 1 / 12,
    res_lon: float = 1 / 12,
) -> xr:
    points = np.column_stack((ds["lon"], ds["lat"]))
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    polygon = np.array(vertices)
    path = matplotlib.path.Path(polygon)

    # Create all points within area
    new_lon = np.arange(x_range[0], x_range[1], res_lon)
    new_lat = np.arange(y_range[0], y_range[1], res_lat)
    xv, yv = np.meshgrid(new_lon, new_lat, indexing="xy")
    points_in_area = np.hstack((xv.reshape((-1, 1)), yv.reshape(-1, 1)))
    garbage_mask = path.contains_points(points_in_area)
    garbage_mask.shape = xv.shape
    # plt.imshow(garbage_mask, origin="lower")  # ,  # interpolation="nearest")

    # Remapping from mask values to lat, lon
    xarray_masked = xr.Dataset(
        dict(garbage=(["lat", "lon"], garbage_mask)), coords=dict(lon=new_lon, lat=new_lat)
    )

    # Test:
    # xarray_masked["garbage"].plot()

    return xarray_masked


def save_xarray(
    xarray: xr, res_lat: float = 1 / 12, res_lon: float = 1 / 12, filename: Optional[AnyStr] = None
) -> None:
    if filename is None:
        filename = f"data/garbage_patch/garbage_patch_res_{res_lat:.3f}_{res_lon:.3f}.nc"
    xarray.to_netcdf(filename)
    print("Saved garbage patch map")


if __name__ == "__main__":
    ds = create_xarray_from_csv()
    ds_map = create_xarray_mask(ds, [-175, -100], [15, 45])
    ds_map["garbage"].plot()
    # save_xarray(ds_map)

    # # # Plot a convex hull around garbage patch
    # ax = set_up_geographic_ax()
    # points = np.column_stack((ds["lon"], ds["lat"]))
    # plot_convex_hull(points, ax)
    # ax = set_up_geographic_ax()
    # plot_alpha_complex(points, ax, alpha=2)
