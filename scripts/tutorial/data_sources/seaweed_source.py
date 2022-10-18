import datetime
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.utils import units

#% Initialze the solar source
casadi_cache_dict = {"deg_around_x_t": 1, "time_around_x_t": 3600 * 24 * 3}
solar_source_dict = {
    "field": "SolarIrradiance",
    "source": "analytical",
    "casadi_cache_settings": casadi_cache_dict,
    "source_settings": {
        "boundary_buffers": [0.2, 0.2],
        "x_domain": [-180, 180],
        "y_domain": [-90, 90],
        "temporal_domain": [
            datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 1, 10, 0, 0, 0, tzinfo=datetime.timezone.utc),
        ],
        "spatial_resolution": 0.1,
        "temporal_resolution": 3600,
    },
}

from ocean_navigation_simulator.data_sources.SolarIrradiance.SolarIrradianceSource import (
    AnalyticalSolarIrradiance,
)

solar_source = AnalyticalSolarIrradiance(solar_source_dict)
# solar_source.update_casadi_dynamics(platform_state) # Only if we need caching but

#%%Instantiate the Seaweed Growth Field -> this takes ≈40 seconds
from ocean_navigation_simulator.data_sources.SeaweedGrowthField import SeaweedGrowthField

seaweed_source_dict = {
    "field": "SeaweedGrowth",
    "source": "GEOMAR",
    "source_settings": {
        "filepath": "./data/nutrients/2021_monthly_nutrients_and_temp.nc",
        "solar_source": solar_source,
    },
}
seaweed_field = SeaweedGrowthField(
    casadi_cache_dict=casadi_cache_dict, hindcast_source_dict=seaweed_source_dict
)

#%% Plot growth potential without irradiance at a specific time over full field
to_plot_time = datetime.datetime(
    year=2021, month=11, day=20, hour=19, minute=0, second=0, tzinfo=datetime.timezone.utc
)
seaweed_field.hindcast_data_source.plot_R_growth_wo_Irradiance(to_plot_time)
#%% The growth at the platform_state if the sun shines as at that time for 12h in percent
platform_state = PlatformState(
    date_time=datetime.datetime(
        year=2021, month=11, day=20, hour=19, minute=0, second=0, tzinfo=datetime.timezone.utc
    ),
    lon=units.Distance(deg=-50.7),
    lat=units.Distance(deg=-44.2),
    seaweed_mass=units.Mass(kg=100),
    battery_charge=units.Energy(watt_hours=100),
)
seaweed_field.hindcast_data_source.F_NGR_per_second(
    platform_state.to_spatio_temporal_casadi_input()
) * 12 * 3600
#%% get the data over a spatio-temporal subset (e.g. for hj_reachability or when we implement caching)
x_interval = [-150, 150]
y_interval = [-60, 60]
t_interval = [
    datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2021, 1, 1, 23, 0, 0, tzinfo=datetime.timezone.utc),
]

subset = seaweed_field.hindcast_data_source.get_data_over_area(
    x_interval, y_interval, t_interval, spatial_resolution=5, temporal_resolution=3600 * 6
)
#%% Plot the F_NGR_per_second at time index 0
import matplotlib.pyplot as plt

subset["F_NGR_per_second"].isel(time=0).plot()
plt.show()
