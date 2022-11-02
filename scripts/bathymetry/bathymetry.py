import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr


def format_to_equally_spaced_xy_grid(xarray):
    """Helper Function to format an xarray to equally spaced lat, lon axis."""
    xarray["lon"] = np.linspace(
        xarray["lon"].data[0], xarray["lon"].data[-1], len(xarray["lon"].data)
    )
    xarray["lat"] = np.linspace(
        xarray["lat"].data[0], xarray["lat"].data[-1], len(xarray["lat"].data)
    )
    return xarray


def format_spatial_resolution(
    xarray: xr.DataArray(), res_lat=1 / 12, res_lon=1 / 12, sample_array=None
):
    if sample_array:
        # Do it like sample array
        raise NotImplementedError()

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


def format_spatial_resolution_global(xarray: xr.DataArray(), res_lat=1 / 12, res_lon=1 / 12):

    # Create a downsampled xarray with zero values
    new_lon = np.arange(xarray.lon[0], xarray.lon[-1], res_lon)
    new_lat = np.arange(xarray.lat[0], xarray.lat[-1], res_lat)

    xarray_downsampled = xr.Dataset(
        dict(elevation=(["lat", "lon"], np.zeros((len(new_lat), len(new_lon))))),
        coords=dict(lon=new_lon, lat=new_lat),
    )
    # Slice the global xarray into 36 slices due to memory limitations (GEBCO_2022.nc is 28GB when loaded)

    # TODO: replace 30 with 180 again
    for i in range(-180, -160, 10):
        # Interpolate each slice and add to downsampled xarray
        print(i)
        interpolated_slice = format_spatial_resolution(xarray.sel(lon=slice(i, i + 10)))
        xarray_downsampled = xr.combine_by_coords(xarray_downsampled, interpolated_slice)
        # xarray_downsampled.assign(interpolated_slice)

        # TODO: debug:
        # .update yields array of Nans
        # combine_by_coords gives some string error
        # xarray_downsampled["elevation"][:, i : i + 10] = format_spatial_resolution(
        #     xarray.sel(lon=slice(i, i + 10))
        # ).to_array()
    return xarray_downsampled


# 3d plot
# TODO: Needs adjustment of aspects
def plot_bathymetry_3d(f, plot_sealevel=False):

    x = f.variables["lon"]
    y = f.variables["lat"]
    z = f.variables["elevation"]

    if plot_sealevel:
        sea_level = np.zeros_like(z)

        fig = go.Figure(
            data=[go.Surface(z=z, x=x, y=y), go.Surface(z=sea_level, x=x, y=y, opacity=0.9)]
        )
    else:
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
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


def plot_bathymetry_2d_cmap(f):
    f["elevation"].plot(cmap="jet")
    plt.show()


def plot_bathymetry_2d_levels(f, depth_decomposition=-1500, depth_min=-30):
    # Depth map with levels set
    f["elevation"].plot(levels=[depth_decomposition, depth_min])
    plt.show()


def plot_bathymetry_orthographic(
    f, lat_center=20, lon_center=-150, depth_decomposition=-1500, depth_min=-30
):
    # Plot elevation on world map with gray background
    p = f["elevation"].plot(
        levels=[depth_decomposition, depth_min],
        subplot_kws=dict(projection=ccrs.Orthographic(lon_center, lat_center), facecolor="gray"),
        transform=ccrs.PlateCarree(),
    )
    p.axes.set_global()
    p.axes.coastlines()
    plt.show()


def generate_global_bathymetry_maps(filename, res_lat, res_lon):
    # , depth_min, depth_decomposition):
    f_original = xr.open_dataset(filename)
    f = format_spatial_resolution_global(f_original)
    # TODO: remove again
    plot_bathymetry_2d_cmap(f)

    # TODO: represent res_lon, res_lat as well
    f.to_netcdf("data/bathymetry/bathymetry_global_downsampled.nc")


if __name__ == "__main__":
    # Load netcdf file
    # file = (
    #     "data/bathymetry/GEBCO_11_Oct_2022_b9b140021f06/gebco_2022_n24.0_s18.0_w-163.0_e-153.0.nc"
    # )
    gebco_global_fname = "data/bathymetry/GEBCO_2022.nc"
    generate_global_bathymetry_maps(gebco_global_fname, 1 / 12, 1 / 12)

    # f = xr.open_dataset(gebco_global_fname)
    # y_range = [f.variables["lat"].data[0], f.variables["lat"].data[-1]]
    # x_range = [f.variables["lon"].data[0], f.variables["lon"].data[-1]]
    # print(f)
    # print(y_range, x_range)
    # print(f.sel(lat=slice(20, 21)))

    # f = format_spatial_resolution(f)
    # # print(f_new_res)

    # depth_min = -30
    # depth_decomposition = -1500

    # # convert to binary
    # area_forbidden = f.variables["elevation"] > depth_min
    # area_floatable = np.logical_not(f.variables["elevation"])
    # area_decomposition = f.variables["elevation"] < -1500

    # res_lat = 1 / 12
    # res_lon = 1 / 12
    # gebco_global_fname = "data/bathymetry/GEBCO_2022.nc"
    # # generate_global_bathymetry_maps(
    # #     gebco_global_fname, res_lat, res_lon
    # # )  # , depth_min, depth_decomposition)

    # f = xr.open_dataset(gebco_global_fname)
    # print(f)
    # print(len(f.variables["lat"]))

    # print("Done")
