import cartopy.crs as ccrs
import cmocean
import dask
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm


def coarsen(
    filename: str, res_lat: float = 1 / 12, res_lon: float = 1 / 12, op="max"
) -> xr.Dataset():
    """Coarsen the global bathymetry map to lower resolution.

    Args:
        filename (str): Global gebco bathymetry map path
        res_lat (float, optional): Target resolution lat in degree. Defaults to 1/12.
        res_lon (float, optional): Target resolution lon in degree. Defaults to 1/12.
        op (str, optional): Operation of coarsening. Defaults to "max".

    Raises:
        NotImplementedError: If operation that is not defined is passed.

    Returns:
        xr.Dataset: Coarsened dataset
    """
    ds = xr.open_dataset(filename, chunks={"lat": 8640, "lon": 4320})
    ds["elevation"] = ds["elevation"].astype("float32")
    # Resolution gebco = 1/240 th degree = 1/4 min
    # To get to e.g. 1/12 degree resolution, we need to combine 1/12 * 240 = 20 points in one
    # TODO: how to inform about rounding in case we need to round?
    lat_coarsening = int(np.rint(res_lat * 240))
    lon_coarsening = int(np.rint(res_lon * 240))
    # Suppress dask warning about chunking size
    print("Starting to coarsen the dataset. This may take a minute.")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        if op == "mean":
            coarsened = ds.coarsen(lat=lat_coarsening, lon=lon_coarsening, boundary="exact").mean()
        elif op == "max":
            coarsened = ds.coarsen(lat=lat_coarsening, lon=lon_coarsening, boundary="exact").max()
        elif op == "min":
            coarsened = ds.coarsen(lat=lat_coarsening, lon=lon_coarsening, boundary="exact").min()
        else:
            raise NotImplementedError("Unknown operation")
    coarsened = coarsened.compute()
    return coarsened


def generate_global_bathymetry_maps(
    filename: str, res_lat: float = 1 / 12, res_lon: float = 1 / 12, op="max"
) -> None:
    """Generate and save global bathymetry map in lower resolution.

    Args:
        filename (str): Global gebco bathymetry map path
        res_lat (float, optional): Target resolution lat in degree. Defaults to 1/12.
        res_lon (float, optional): Target resolution lon in degree. Defaults to 1/12.
        op (str, optional): Operation of coarsening. . Defaults to "max".
    """
    low_res = coarsen(filename, res_lat, res_lon, op)
    low_res.to_netcdf(f"data/bathymetry/bathymetry_global_res_{res_lat:.3f}_{res_lon:.3f}_{op}.nc")
    print("Saved global bathymetry map with lower resolution")


def generate_shortest_distance_maps(xarray: xr, elevation: float = 0) -> xr:

    # Will be converted from dataset to dataarray
    # 0 land, 1 water
    xarray_land = xr.where(xarray["elevation"] > elevation, 0, 10000)
    data = xarray_land.data
    lat = xarray_land.coords["lat"].values
    lon = xarray_land.coords["lon"].values
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    min_d_map = bfs_min_distance(data, lat, lon)
    print("hi")


def naive_min_distance(data, lat, lon):

    # Create
    distance = np.full_like(data, 10000000)

    # Prepare for haversine
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    land_lat_lon_list = []
    # Construct (n,2) shaped array of [(lat, lon), ]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 0:
                land_lat_lon_list.append([lat[i], lon[j]])
    land_lat_lon = np.array(land_lat_lon_list)
    land_lat_lon_rad = np.deg2rad(land_lat_lon)

    # O(n^4, not parallel), very naive
    for i in tqdm(range(data.shape[0])):
        for j in tqdm(range(data.shape[1]), leave=False):
            # d_min = 100000000
            # Skip land
            if data[i][j] == 0:
                continue

            # Would take > day...
            # for ii in range(data.shape[0]):
            #     for jj in range(data.shape[1]):
            #         if data[ii][jj] != 0:
            #             continue
            #         # Skip self
            #         if ii == i and jj == j:
            #             continue
            #         d_min = min(
            #             d_min, haversine_distances([[lat[i], lon[j]]], [[lat[ii], lon[jj]]])
            #         )
            d_min = np.min(haversine_distances([[lat[i], lon[j]]], land_lat_lon_rad))
            distance[i][j] = d_min

    distance *= 6371000 / 1000  # Convert to kilometers
    # TODO: plot

    # Next one: do n2 to construct land lat, lon array, then do n2 of haversine to land. Haversine will be parallel


def bfs_min_distance(data, lat, lon, connectivity=4):

    if connectivity == 4:
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    elif connectivity == 8:
        dirs = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    else:
        raise NotImplementedError()

    min_d_map = data.astype(float)
    visited = set()
    from collections import deque

    q = deque()
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            if min_d_map[i][j] == 0:  # land
                q.append((i, j))

    while q:
        # First iteration will do all land, next ones will do all coastline, then all coastline + 1 field away
        for idx in tqdm(range(len(q))):
            pos = q.popleft()
            i, j = pos
            visited.add(pos)

            distances = []
            for dir in dirs:
                i_neighbor, j_neighbor = i + dir[0], j + dir[1]

                # Add not visited neighboring sea cells to deque
                # Add distance to nearest land
                if (
                    i_neighbor >= 0
                    and i_neighbor < data.shape[0]
                    and j_neighbor >= 0
                    and j_neighbor < data.shape[1]
                ):
                    # Add sea neighbors
                    if (i_neighbor, j_neighbor) not in visited and min_d_map[i_neighbor][
                        j_neighbor
                    ] != 0:
                        # TODO: check if lookup in deque reduces speed by 80%
                        if (i_neighbor, j_neighbor) not in q:
                            q.append((i_neighbor, j_neighbor))

                    # Skip finding land if we are on land
                    if min_d_map[i][j] == 0:
                        continue
                    distances.append(
                        haversine_distances(
                            [[lat[i], lon[j]]], [[lat[i_neighbor], lon[j_neighbor]]]
                        )[0][0]
                        + min_d_map[i_neighbor][j_neighbor]
                    )
            if min_d_map[i][j] != 0:
                min_d_map[i][j] = min(distances)
    min_d_map *= 6371000 / 1000  # Convert to kilometers
    return min_d_map


def format_spatial_resolution(xarray: xr.Dataset(), res_lat=1 / 12, res_lon=1 / 12) -> xr.Dataset():
    """Format the spatial resolution.

    Args:
        filename (str): Global gebco bathymetry map path
        res_lat (float, optional): Target resolution lat in degree. Defaults to 1/12.
        res_lon (float, optional): Target resolution lon in degree. Defaults to 1/12.
    Returns:
        xr.Dataset: Reduced resolution dataset
    """

    new_lon = np.arange(xarray.lon[0], xarray.lon[-1], res_lon)
    new_lat = np.arange(xarray.lat[0], xarray.lat[-1], res_lat)
    xarray_new = xarray.interp(
        lat=new_lat,
        lon=new_lon,
        # Extrapolate to avoid nan values as edges
        kwargs={
            "fill_value": "extrapolate",
        },
    )
    return xarray_new


def plot_bathymetry_2d(xarray: xr.Dataset()) -> None:
    """Generate 2d plot with colorscale."""
    xarray["elevation"].plot()
    plt.show()


def plot_bathymetry_2d_levels(
    xarray: xr.Dataset(), depth_decomposition: float = -1500, depth_min: float = -150
) -> None:
    """Generate 2d plot with only 3 levels: deep enough for decomposition,
    deep enough for navigation, but not decomposition, too shallow.

    Args:
        xarray (xr.Dataset): Input bathymetry map
        depth_decomposition (float, optional): Minimal depth for decomposition. Defaults to -1500.
        depth_min (float, optional): Minimal depth we want to be in. Defaults to -150.
    """
    xarray["elevation"].plot(levels=[depth_decomposition, depth_min])
    plt.show()


def plot_bathymetry_orthographic(
    xarray: xr.Dataset(), lat_center=20, lon_center=-150, depth_decomposition=-1500, depth_min=-150
):
    """Plot bathymetry in 2d with orthographic projection.

    Args:
        xarray (xr.Dataset): Input bathymetry map
        lat_center (int, optional): Center of view lat. Defaults to 20.
        lon_center (int, optional): Center of view lon. Defaults to -150.
        depth_decomposition (float, optional): Minimal depth for decomposition. Defaults to -1500.
        depth_min (float, optional): Minimal depth we want to be in. Defaults to -150.
    """
    p = xarray["elevation"].plot(
        levels=[depth_decomposition, depth_min],
        subplot_kws=dict(projection=ccrs.Orthographic(lon_center, lat_center), facecolor="gray"),
        transform=ccrs.PlateCarree(),
    )
    p.axes.set_global()
    p.axes.coastlines()
    plt.show()


def plot_bathymetry_3d(xarray: xr.Dataset(), plot_sealevel: bool = False):
    """Generate interactive 3d plot in browser."""
    x = xarray.variables["lon"]
    y = xarray.variables["lat"]
    z = xarray.variables["elevation"]
    colorscale = mpl_to_plotly(cmocean.cm.delta)
    if plot_sealevel:
        sea_level = np.zeros_like(z)

        fig = go.Figure(
            data=[go.Surface(z=z, x=x, y=y), go.Surface(z=sea_level, x=x, y=y, opacity=0.9)]
        )
    else:
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale=colorscale, cmid=0)])
    fig.update_traces(
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    )
    # Reverse latitude orientation
    fig.update_layout(
        title="Elevation",
        autosize=False,
        width=900,
        height=900,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Elevation",
            # xaxis={"autorange": "reversed"},
            # aspectmode="cube",
        ),
        margin=dict(r=20, b=10, l=10, t=10),
    )
    # fig.update_layout(color='Elevation')
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Elevation [m]",
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=200,
            yanchor="top",
            y=1,
            ticks="outside",
            ticksuffix="",
            dtick=5,
        )
    )
    fig.show()


def mpl_to_plotly(cmap: plt.cm, pl_entries: int = 11, rdigits: int = 2):
    """Convert matplotlib colorscale to plotly

    Args:
        cmap (plt.cm): Matplotlib colormap
        pl_entries (int, optional): Number of Plotly colorscale entries. Defaults to 11.
        rdigits (int, optional): Number of digits for rounding scale values. Defaults to 2.

    Returns:
        _type_: _description_
    """
    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3] * 255).astype(np.uint8)
    pl_colorscale = [[round(s, rdigits), f"rgb{tuple(color)}"] for s, color in zip(scale, colors)]
    return pl_colorscale


if __name__ == "__main__":
    # Generate global bathymetrymap and save to disk
    # gebco_global_filename = "data/bathymetry/GEBCO_2022.nc"
    # generate_global_bathymetry_maps(gebco_global_filename)

    # Test all other functions
    low_res_loaded = xr.open_dataset("data/bathymetry/bathymetry_global_res_0.083_0.083_max.nc")
    # plot_bathymetry_orthographic(low_res_loaded)
    # plot_bathymetry_2d_levels(low_res_loaded)
    # plot_bathymetry_2d(low_res_loaded)
    # Low res is still too large for 3d interactive plot
    very_low_res = format_spatial_resolution(low_res_loaded, res_lat=1, res_lon=1)
    # plot_bathymetry_3d(very_low_res)

    # low_res_loaded = xr.open_dataset("data/bathymetry/bathymetry_global_res_0.083_0.083_max.nc")
    min_d_map = generate_shortest_distance_maps(very_low_res)
    plt.imshow(min_d_map, origin="lower")
