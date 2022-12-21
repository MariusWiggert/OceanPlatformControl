import datetime

import cmocean
import matplotlib.pyplot as plt
import xarray as xr

from ocean_navigation_simulator.data_sources.OceanCurrentField import (
    OceanCurrentField,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.utils import units
from scripts.safety.bathymetry import plot_rectangles_from_interval


def plot_currents_in_area(area: str):
    casadi_cache_dict = {"deg_around_x_t": 1, "time_around_x_t": 3600 * 24 * 1}
    hindcast_source_dict = {
        "field": "OceanCurrents",
        "source": "hindcast_files",
        "source_settings": {"folder": "data/Copernicus/Hindcast/Region1/"},
    }
    forecast_source_dict = None
    ocean_field = OceanCurrentField(
        hindcast_source_dict=hindcast_source_dict,
        forecast_source_dict=forecast_source_dict,
        casadi_cache_dict=casadi_cache_dict,
    )

    t_0 = datetime.datetime(2022, 10, 10, 13, 00, tzinfo=datetime.timezone.utc)

    if area == "full":
        # Full
        x_interval = [-160, -105]
        y_interval = [15, 40]
        output_name = "region1_full_hindcast_current_animation.mp4"
    elif area == "sf":
        # SF
        x_interval = [-125, -121]
        y_interval = [36, 40]
        output_name = "region1_sf_hindcast_current_animation.mp4"
    elif area == "la":
        # LA
        x_interval = [-121, -117]
        y_interval = [32, 35]
        output_name = "region1_la_hindcast_current_animation.mp4"
    elif area == "hawaii":
        # Hawaii -> Just outside of this, but still in region 1
        x_interval = [-160, -152]
        y_interval = [18, 23]
        output_name = "region1_hawaii_hindcast_current_animation.mp4"
    elif area == "bajaNE":
        # Baja california, east side channel
        x_interval = [-114, -111]
        y_interval = [28, 30]
        output_name = "region1_bajaNE_hindcast_current_animation.mp4"
    elif area == "puertoVallarta":
        # Puerto Vallarta, Mexico mainland south east of baja california
        x_interval = [-107, -105]
        y_interval = [19, 22]
        output_name = "region1_puertoVallarta_hindcast_current_animation.mp4"
    # x_0 = PlatformState(lon=units.Distance(deg=-130), lat=units.Distance(deg=20), date_time=t_0)
    # x_T = SpatialPoint(lon=units.Distance(deg=-140), lat=units.Distance(deg=35))
    t_interval = [t_0, t_0 + datetime.timedelta(days=5)]
    kwargs = {"vmax": 1, "vmin": 0}

    ax = ocean_field.hindcast_data_source.plot_data_at_time_over_area(
        time=t_0,
        x_interval=x_interval,
        y_interval=y_interval,
        plot_type="streamline",
        # plot_type="quiver",
        return_ax=True,
        **kwargs
    )
    folder_to_save_in = "generated_media/" if not output_name.startswith("/") else ""
    plt.savefig(folder_to_save_in + output_name[:-3] + "png")

    ocean_field.hindcast_data_source.animate_data(
        x_interval=x_interval,
        y_interval=y_interval,
        t_interval=t_interval,
        output=output_name,
        **kwargs
    )


def plot_area_with_rectangles(map_type: str):
    if map_type == "bathymetry":
        xarray = xr.open_dataset("data/bathymetry/bathymetry_global_res_0.083_0.083_max.nc")
        p = xarray["elevation"].plot(cmap=cmocean.cm.delta, vmin=-6000, vmax=6000)

    # t_0 = datetime.datetime(2022, 10, 10, 13, 00, tzinfo=datetime.timezone.utc)

    # Full
    x_interval = [-160, -105]
    y_interval = [15, 40]
    ax = plot_rectangles_from_interval(x_interval, y_interval, p.axes, color="r")

    # SF
    x_interval = [-125, -121]
    y_interval = [36, 40]
    ax = plot_rectangles_from_interval(x_interval, y_interval, p.axes)

    # LA
    x_interval = [-121, -117]
    y_interval = [32, 35]
    ax = plot_rectangles_from_interval(x_interval, y_interval, p.axes)

    # Hawaii -> Just outside of this, but still in region 1
    x_interval = [-160, -152]
    y_interval = [18, 23]
    ax = plot_rectangles_from_interval(x_interval, y_interval, p.axes)

    # Baja california, east side channel
    x_interval = [-114, -111]
    y_interval = [28, 30]
    ax = plot_rectangles_from_interval(x_interval, y_interval, p.axes)

    # Puerto Vallarta, Mexico mainland south east of baja california
    x_interval = [-107, -105]
    y_interval = [19, 22]
    ax = plot_rectangles_from_interval(x_interval, y_interval, p.axes)
    plt.show()


if __name__ == "__main__":
    map_type = "bathymetry"
    # plot_area_with_rectangles(map_type)

    areas = ["sf", "la", "hawaii", "bajaNE", "puertoVallarta"]
    for area in areas:
        plot_currents_in_area(area=area)
