from pathlib import Path
from typing import List

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import ConvexHull, Delaunay


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
    plt.text(270.5, 459, f"alpha={alpha}", size=18)
    plt.show()


def create_xarray_from_csv(
    filename: str = "data/lebreton/Lebreton2018_HistoricalDataset.csv",
) -> xr:
    """Create an xarray from a csv with data

    Args:
        filename (str, optional): Csv file containing garbage patch data. Defaults to "data/lebreton/Lebreton2018_HistoricalDataset.csv".

    Returns:
        xr: Xarray with points within the GPGP (great pacific garbage patch).
    """
    df = pd.read_csv(filename)
    df = df.rename(columns={"Latitude (degrees)": "lat", "Longitude (degrees)": "lon"})
    df = df[df["Inside GPGP?"] == 1]
    ds = xr.Dataset.from_dataframe(df)
    return ds


def create_xarray_around_points(
    lat: List[float],
    lon: List[float],
    radius_around_points: float = 1,
    num_points_per_circle: int = 20,
) -> xr:
    """Create xarray with fake points within a circle around all input coordinate pairs.

    Args:
        lat (List[float]): List of latitude coordinate centers of circles
        lon (List[float]): List of longitude coordinate centers of circles
        radius_around_points (float, optional): Radius around points to generate points in, in degree. Defaults to 1.
        num_points_per_circle (int, optional): Amount of points per to generate per circle. Defaults to 20.

    Returns:
        xr: Xarray with points of garbage
    """
    lat *= num_points_per_circle
    lon *= num_points_per_circle
    lat += np.random.uniform(-radius_around_points, radius_around_points, len(lat))
    lon += np.random.uniform(-radius_around_points, radius_around_points, len(lon))
    df = pd.DataFrame({"lat": lat, "lon": lon})
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
    algorithm: str = "ConvexHull",
    alpha: float = 2.0,
) -> xr:
    """Create an xarray mask of garbage patches

    Args:
        xarray (xr): xarray containing lat, lon columns of positions within a garbage patch.
        x_range (List[float]): List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
        y_range (List[float]): List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
        res_lat (float, optional): Target resolution lat in degree. Defaults to 1/12.
        res_lon (float, optional): Target resolution lat in degree. Defaults to 1/12.
        algorithm (str, optional): Select which algorithm to use to get perimeter around points. Defaults to "ConvexHull".
        alpha (float, optional): Alpha value if AlphaComplex is chosen as algorithm. Defaults to 2.0.

    Raises:
        NotImplementedError: If you select an algorithm that is not supported.

    Returns:
        xr: Xarray with lat, lon, garbage fields with boolean mask of garbage. True for garbage.
    """
    points = np.column_stack((ds["lon"], ds["lat"]))

    # Select algorithm
    if algorithm == "ConvexHull":
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
    elif algorithm == "AlphaComplex":
        edges = alpha_complex(points, alpha)
        edges_list = [pair for pair in edges]
        vertices_indices = [edges_list[0][0]]

        i = 0
        while len(edges_list):
            # TODO: this is if there are multiple unconnected alpha complexes. Possible need to debug alpha_complexes
            i += 1
            if i > 1000:
                print("Failure to create connected alpha complexes.")
                break
            for pair in edges_list:
                if vertices_indices[-1] in pair:
                    if vertices_indices[-1] == pair[0]:
                        vertices_indices.append(pair[1])
                    elif vertices_indices[-1] == pair[1]:
                        vertices_indices.append(pair[0])
                    edges_list.remove(pair)
                else:
                    continue

        vertices = points[vertices_indices]
        # raise NotImplementedError("AlphaComplex WIP, select Complex hull")
    else:
        raise NotImplementedError("Select algorithm that is implemented")

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
    return xarray_masked


def save_xarray(xarray: xr, filename: str) -> None:
    """Save xarray to disk

    Args:
        xarray (xr): xarray
        filename (str): filename with path, dirs will be created if not existing.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    xarray.to_netcdf(filename)
    print(f"Saved garbage patch mask at {filename}")


if __name__ == "__main__":
    # region_1 = [[-175, -100], [15, 45]]

    # region_global = [[-180, 180], [-90, 90]]
    # res_lat = 1 / 12
    # res_lon = 1 / 12
    # ds = create_xarray_from_csv()
    # xarray = create_xarray_mask(ds, *region_global, res_lat=res_lat, res_lon=res_lon)
    # save_xarray(
    #     xarray, f"data/garbage_patch/garbage_patch_global_res_{res_lat:.3f}_{res_lon:.3f}.nc"
    # )

    # xarray["garbage"].plot()
    # plt.show()

    # Global with fake garbage in gulf of mexico
    region_global = [[-180, 180], [-90, 90]]
    res_lat = 1 / 12
    res_lon = 1 / 12
    lat = [26, 26, 26]
    lon = [-87, -86, -85]
    ds = create_xarray_around_points(lat, lon)
    xarray = create_xarray_mask(ds, *region_global, res_lat=res_lat, res_lon=res_lon)
    save_xarray(
        xarray,
        f"data/garbage_patch/garbage_patch_fake_gulf_of_mexico_global_res_{res_lat:.3f}_{res_lon:.3f}.nc",
    )

    xarray["garbage"].plot()
    plt.show()

    # # # Plot a convex hull around garbage patch
    # ax = set_up_geographic_ax()
    # points = np.column_stack((ds["lon"], ds["lat"]))
    # plot_convex_hull(points, ax)
    # ax = set_up_geographic_ax()
    # plot_alpha_complex(points, ax, alpha=2)
