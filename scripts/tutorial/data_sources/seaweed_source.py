import datetime

import matplotlib.pyplot as plt

from ocean_navigation_simulator.data_sources.SeaweedGrowth.SeaweedGrowthSource import SeaweedGrowthCali, \
    SeaweedGrowthGEOMAR, SeaweedGrowthCircles
from ocean_navigation_simulator.data_sources.SeaweedGrowthField import SeaweedGrowthField
from ocean_navigation_simulator.data_sources.SolarIrradianceField import SolarIrradianceField
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.utils import units

casadi_cache_dict = {"deg_around_x_t": 1, "time_around_x_t": 3600 * 24 * 1}
# %% Set up solar field as base data source
# Step 1: create the specification dict
source_dict = {
    "field": "SolarIrradiance",
    "source": "analytical_wo_caching",  # can also be analytical_w_caching
    "source_settings": {
        "boundary_buffers": [0.2, 0.2],
        "x_domain": [-180, 180],
        "y_domain": [-90, 90],
        "temporal_domain": [
            datetime.datetime(2020, 1, 1, 0, 0, 0),
            datetime.datetime(2024, 1, 10, 0, 0, 0),
        ],
        "spatial_resolution": 0.1,
        "temporal_resolution": 3600,
    },
}
# % Step 2: Instantiate the field
solar_field = SolarIrradianceField(
    hindcast_source_dict=source_dict, casadi_cache_dict=casadi_cache_dict
)

# %% GEOMAR
seaweed_dict = {
    'field': 'SeaweedGrowth',
    'source': 'GEOMAR',
    'source_settings': {
        'filepath': './ocean_navigation_simulator/package_data/nutrients/',
        'solar_source': solar_field.hindcast_data_source},
    'use_geographic_coordinate_system': True
}

# %% California
seaweed_dict = {
    'field': 'SeaweedGrowth',
    'source': 'California',
    'source_settings': {
        'filepath': './ocean_navigation_simulator/package_data/cali_growth_map/',
        'solar_source': solar_field.hindcast_data_source,
        'max_growth': 0.2,
        'respiration_rate': 0.01,
    },
    'use_geographic_coordinate_system': True
}

# %% Analytical with circles
seaweed_dict = {
    'field': 'SeaweedGrowth',
    'source': 'SeaweedGrowthCircles',
    'source_settings': {
        # Specific for it
        'cirles': [[9, 2.5, 0.5], [1, 8, 0.2]],  # [x, y, r]
        'NGF_in_time_units': [0.5, 1.0],  # [NGF]
        # Just boundary stuff
        'boundary_buffers': [0., 0.],
        'x_domain': [0, 10],
        'y_domain': [0, 10],
        'temporal_domain': [0, 100],
        'spatial_resolution': 0.1,
        'temporal_resolution': 10, }
}

# %% Instantiate the source
SeaweedSource = SeaweedGrowthField.instantiate_source_from_dict(seaweed_dict)

# %% Test Data for GEOMAR
t_0 = datetime.datetime(2023, 4, 11, 0, 0, 0, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=3)]
x_interval = [-170, +170]
y_interval = [-80, +80]
x_0 = PlatformState(lon=units.Distance(deg=-120), lat=units.Distance(deg=30), date_time=t_0)
x_T = SpatialPoint(lon=units.Distance(deg=-122), lat=units.Distance(deg=37))

# %% Test Data for California
t_0 = datetime.datetime(2023, 4, 11, 0, 0, 0, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=3)]
x_interval = [-124, -121.5]
y_interval = [36, 38.98]
x_0 = PlatformState(lon=units.Distance(deg=-123), lat=units.Distance(deg=36.8), date_time=t_0)
x_T = SpatialPoint(lon=units.Distance(deg=-122), lat=units.Distance(deg=37))

#%% debug
SeaweedSource.DataArray["R_growth_wo_Irradiance"].isel(time=0).plot()
plt.show()
# %% First test: get data at point
growth = SeaweedSource.get_data_at_point(spatio_temporal_point=x_0)
print("Seaweed Growth at x_0 is {}% per day".format(growth * 100 * 3600 * 24))
# %% Plot it at time over area
SeaweedSource.plot_data_at_time_over_area(
    time=t_0, x_interval=x_interval, y_interval=y_interval,
)
# %% Animate it over time
SeaweedSource.animate_data(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    # spatial_resolution=5,
    temporal_resolution=3600 * 1,
    output="seaweed_test_animation.mp4",
)
# %% check it for a spatio-temporal area
area_xarray = SeaweedSource.get_data_over_area(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    # spatial_resolution=1,
    temporal_resolution=3600 * 3,
)
# %% check if it for California Coast
# slice the data to the california coast
area_xarray.sel(lat=slice(30, 50), lon=slice(-130, -110)).isel(time=0)['F_NGR_per_second'].plot()
plt.show()

#%% Note: Looks decent, but somehow the solar source might be weird in this area.
# => render solar source animation for the area. Might also be the transformation from solar source to irradiance rate.
#%% let's plot just the Irradiance Factor =)