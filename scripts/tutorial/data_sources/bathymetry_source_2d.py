import datetime

import matplotlib.pyplot as plt

from ocean_navigation_simulator.data_sources.Bathymetry.BathymetrySource import (
    BathymetrySource2d,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.utils.units import Distance

# Initialize bathymetry source
bathymetry_source_dict = {
    "field": "Bathymetry",
    "source": "gebco",
    "source_settings": {
        "filepath": "bathymetry_global_res_0.083_0.083_max.nc"
    },
    "distance": {
        "filepath": "bathymetry_distance_res_0.004_max_elevation_0_northern_california.nc",
        "safe_distance": 0.01,
    },
    "casadi_cache_settings": {"deg_around_x_t": 20},
    "use_geographic_coordinate_system": True,
}
bathymetry_field = BathymetrySource2d(
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
#%% Plot bathymetry over full field
ax = bathymetry_field.plot_data_over_area(
    bathymetry_field.grid_dict["x_range"], bathymetry_field.grid_dict["y_range"]
)
plt.show()
# Plot over region 1
ax = bathymetry_field.plot_data_over_area([-160, -105], [15, 40])
plt.show()
