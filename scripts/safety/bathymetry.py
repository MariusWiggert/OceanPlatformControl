import cartopy.crs as ccrs
import cmocean
import dask
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr


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


# def generate_shortest_distance_maps(map: xr, elevation: float) -> xr:

#     x_range =
#     y_range =
#     if point is > elevation:
#         set value to 0
#     else:
#         val = min( value of neighbor + distance(neighbor, me))


#     pass


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
    colorscale = "Earth"
    # TODO: Cmocean colorscale seems to break something,
    # this is why we use the not so great "Earth" one
    # Here is docu for an older plotly version
    #  https://plotly.com/python/v3/cmocean-colorscales/
    # elem_len = [len(x), len(y), len(z)]
    # colorscale = cmocean_to_plotly(cmocean.cm.delta, max(elem_len))
    if plot_sealevel:
        sea_level = np.zeros_like(z)

        fig = go.Figure(
            data=[go.Surface(z=z, x=x, y=y), go.Surface(z=sea_level, x=x, y=y, opacity=0.9)]
        )
    else:
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale=colorscale)])
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


def cmocean_to_plotly(cmap: cmocean.cm, pl_entries: int):
    """Helper function to adapt cmocean colorbar for plotly.
    Seems to break plotly surface at the moment
    """
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = [*map(np.uint8, np.array(cmap(k * h)[:3]) * 255)]
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])

    return pl_colorscale


if __name__ == "__main__":
    # Generate global bathymetrymap and save to disk
    gebco_global_filename = "data/bathymetry/GEBCO_2022.nc"
    generate_global_bathymetry_maps(gebco_global_filename)

    # Test all other functions
    # low_res_loaded = xr.open_dataset("data/bathymetry/bathymetry_global_res_0.083_0.083_max.nc")
    # plot_bathymetry_orthographic(low_res_loaded)
    # plot_bathymetry_2d_levels(low_res_loaded)
    # plot_bathymetry_2d(low_res_loaded)
    # Low res is still too large for 3d interactive plot
    # very_low_res = format_spatial_resolution(low_res_loaded, res_lat=1 / 4, res_lon=1 / 4)
    # plot_bathymetry_3d(very_low_res)
