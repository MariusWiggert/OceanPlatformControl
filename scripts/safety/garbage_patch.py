import cartopy.crs as ccrs
import cartopy.feature as cfeature
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

    # Plot an alpha complex around points
    for idx in range(1, 4):
        ax = set_up_geographic_ax()
        edges = alpha_complex(points, alpha=idx * 3)
        plt.plot(points[:, 0], points[:, 1], ".")
        for i, j in edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1])
        plt.text(270.5, 459, "alpha=1", size=18)
        plt.show()
