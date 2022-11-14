import datetime

from ocean_navigation_simulator.utils.units import Distance
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)

from ocean_navigation_simulator.data_sources.Bathymetry.BathymetrySource import BathymetrySource
import matplotlib.pyplot as plt

# Initialize bathymetry source
# TODO: check if time_around_x_t is needed
casadi_cache_dict = {"deg_around_x_t": 20, "time_around_x_t": 3600 * 24 * 1}
bathymetry_source_dict = {
    "field": "Bathymetry",
    "source": "gebco",
    "source_settings": {"filepath": "data/bathymetry/bathymetry_global_res_0.083_0.083_max.nc"},
}
# TODO: possibly instead of source_dict use "bathymetry_source_dict"
bathymetry_field = BathymetrySource(
    casadi_cache_dict=casadi_cache_dict,
    source_dict=bathymetry_source_dict,
)


#%% Check if degree around work and if get_data_at_point_work
# Expect inaccuracies due to resolution of bathymetry map
t_0 = datetime.datetime(2021, 11, 25, 23, 30, tzinfo=datetime.timezone.utc)
# Define Mariannas Trench
lat = 11.326344
lon = 142.187248
mariannas_trench = SpatialPoint(lat=Distance(deg=lat), lon=Distance(deg=lon))
x_0 = PlatformState(lon=Distance(deg=lon - 10), lat=Distance(deg=lat + 10), date_time=t_0)
bathymetry_field.update_casadi_dynamics(x_0)
print(f"Elevation at {mariannas_trench} is {bathymetry_field.get_data_at_point(mariannas_trench)}")

# Define Mount Everest
lat = 27.9881
lon = 86.9250
everest = SpatialPoint(lat=Distance(deg=lat), lon=Distance(deg=lon))
x_0 = PlatformState(lon=Distance(deg=lon - 10), lat=Distance(deg=lat + 10), date_time=t_0)
bathymetry_field.update_casadi_dynamics(x_0)
print(f"Elevation at {everest} is {bathymetry_field.get_data_at_point(everest)}")

print(f"Everest is higher than 8000m: {bathymetry_field.is_higher_than(everest, 8000)}")
# We can also check if casadi loaded this area (This should fail)
# print(
#     f"Mariannas Trench is higher than -9000m: {bathymetry_field.is_higher_than(mariannas_trench, -9000)}"
# )

#%% Plot bathymetry over full field
# TODO: possibly need time as required by higher level function
ax = bathymetry_field.plot_data_at_time_over_area(
    0, bathymetry_field.grid_dict["x_range"], bathymetry_field.grid_dict["y_range"]
)
plt.show()
print("Hurray")
