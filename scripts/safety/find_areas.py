import datetime

import numpy as np

from ocean_navigation_simulator.data_sources.OceanCurrentField import (
    OceanCurrentField,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.utils import units

# For fast interpolation of currents we cache part of the spatio-temporal data around x_t in a casadi function
casadi_cache_dict = {"deg_around_x_t": 1, "time_around_x_t": 3600 * 24 * 1}
hindcast_source_dict = {
    "field": "OceanCurrents",
    "source": "hindcast_files",
    "source_settings": {"folder": "data/california/"},
}

forecast_source_dict = None

ocean_field = OceanCurrentField(
    hindcast_source_dict=hindcast_source_dict,
    forecast_source_dict=forecast_source_dict,
    casadi_cache_dict=casadi_cache_dict,
)
# copernicus_hindcast_lon_[-150, -115]_lat_[0, 40]_time_[2022-10-01 12 00 00,2022-10-10 12 00 00]
import pytz

t_0 = datetime.datetime(2022, 10, 1, 12, 30, tzinfo=datetime.timezone.utc)
# Full
# x_interval = [-150, -115] -> -160, -105
# y_interval = [0, 40] -> should be 15, 40

# SF
x_interval = [-125, -121]
y_interval = [36, 40]
t_interval = [t_0, t_0 + datetime.timedelta(days=5)]
output_name = "california_sf_hindcast_current_animation.mp4"

# LA
x_interval = [-121, -117]
y_interval = [32, 35]
t_interval = [t_0, t_0 + datetime.timedelta(days=5)]
output_name = "california_la_hindcast_current_animation.mp4"


# # Hawaii -> Just outside of this, but still in region 1
# x_interval = [-160, -152]
# y_interval = [18, 23]
# t_interval = [t_0, t_0 + datetime.timedelta(days=7)]
# output_name = "california_la_hindcast_current_animation.mp4"

# # Baja california, east side channel
# x_interval = [-114, -111]
# y_interval = [28, 30]
# t_interval = [t_0, t_0 + datetime.timedelta(days=7)]
# output_name = "california_bajaNE_hindcast_current_animation.mp4"


# # Lázaro Cárdenas: double gyre, leads to onshore currents
# x_interval = [-104, -101]
# y_interval = [17, 19]
# t_interval = [t_0, t_0 + datetime.timedelta(days=7)]
# output_name = "california_lazaroCardenas_hindcast_current_animation.mp4"


# # Puerto Vallarta, Mexico mainland south east of baja california
# x_interval = [-107, -105]
# y_interval = [19, 22]
# t_interval = [t_0, t_0 + datetime.timedelta(days=7)]
# output_name = "california_puertoVallarta_hindcast_current_animation.mp4"


x_0 = PlatformState(lon=units.Distance(deg=-130), lat=units.Distance(deg=20), date_time=t_0)
x_T = SpatialPoint(lon=units.Distance(deg=-140), lat=units.Distance(deg=35))


ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=t_0,
    x_interval=x_interval,
    y_interval=y_interval,
    # plot_type='streamline',
    plot_type="quiver",
    return_ax=False,
)

ocean_field.hindcast_data_source.animate_data(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    output=output_name,
)
