import datetime
import matplotlib.pyplot as plt

from ocean_navigation_simulator.data_sources.GarbagePatch.GarbagePatchSource import (
    GarbagePatchSource2d,
)
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint

from ocean_navigation_simulator.utils.units import Distance

garbage_patch_source_dict = {
    "field": "Garbage",
    "source": "Lebreton",
    "source_settings": {"filepath": "data/garbage_patch/garbage_patch_region_1_res_0.083_0.083.nc"},
    "casadi_cache_settings": {"deg_around_x_t": 10},
    "use_geographic_coordinate_system": True,
}

garbage_patch_field = GarbagePatchSource2d(source_dict=garbage_patch_source_dict)


##% Check if degree around work and if get data at point work
t_0 = datetime.datetime(2021, 11, 25, 23, 30, tzinfo=datetime.timezone.utc)
lat = 32
lon = -140.494
point_in_gpgp = SpatialPoint(lat=Distance(deg=lat), lon=Distance(deg=lon))
x_0 = PlatformState(lon=Distance(deg=lon), lat=Distance(deg=lat), date_time=t_0)
garbage_patch_field.update_casadi_dynamics(x_0)
print(
    f"Point {point_in_gpgp} is in garbage patch {garbage_patch_field.get_data_at_point(point_in_gpgp)}"
)


# # TODO: investigate outside of file exception
# lat = 22
# lon = -155
# point_outside_gpgp = SpatialPoint(lat=Distance(deg=lat), lon=Distance(deg=lon))
# x_0 = PlatformState(lon=Distance(deg=lon), lat=Distance(deg=lat + 20), date_time=t_0)
# garbage_patch_field.update_casadi_dynamics(x_0)
# print(
#     f"Point {point_outside_gpgp} is in garbage patch {garbage_patch_field.get_data_at_point(point_outside_gpgp==True)}"
# )

#%% Plot garbage patch over full area
ax = garbage_patch_field.plot_data_over_area(
    garbage_patch_field.grid_dict["x_range"], garbage_patch_field.grid_dict["y_range"]
)
plt.show()
# Region 1
ax = garbage_patch_field.plot_data_over_area([-160, -105], [15, 40])
plt.show()
print("Hurray")
