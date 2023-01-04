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

#%% Create the source dict for the ocean current
#%% Option 1: Accessing data in the Copernicus (or HYCOM) server directly via opendap -> data loaded when needed
hindcast_source_dict = {
    "field": "OceanCurrents",
    "source": "opendap",
    "source_settings": {
        "service": "copernicus",
        "currents": "total",
        "USERNAME": "mmariuswiggert",
        "PASSWORD": "tamku3-qetroR-guwneq",
        "DATASET_ID": "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
    },
}
forecast_source_dict = None
#%% Option 2: Accessing data via local files for forecasts and hindcasts
hindcast_source_dict = {
    "field": "OceanCurrents",
    "source": "hindcast_files",
    "source_settings": {"folder": "data/hindcast_test/"},
}

# Adding forecast files if we want!
forecast_source_dict = {
    "field": "OceanCurrents",
    "source": "forecast_files",
    "source_settings": {"folder": "data/forecast_test/"},
}

#%% Create the ocean Field object (containing both the hindcast and optionally the forecast source)
ocean_field = OceanCurrentField(
    hindcast_source_dict=hindcast_source_dict,
    forecast_source_dict=forecast_source_dict,
    casadi_cache_dict=casadi_cache_dict,
)
#%%
import pytz
t_0 = datetime.datetime(2021, 11, 25, 23, 30, tzinfo=datetime.timezone.utc)
#%% Use it by defining a spatial and temporal interval and points
t_0 = datetime.datetime(2021, 11, 25, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=2)]
x_interval = [-82, -80]
y_interval = [24, 26]
x_0 = PlatformState(lon=units.Distance(deg=-81.5), lat=units.Distance(deg=23.5), date_time=t_0)
x_T = SpatialPoint(lon=units.Distance(deg=-80), lat=units.Distance(deg=24.2))
#%% plot ocean currents at time over an area
ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=t_0,
    x_interval=x_interval,
    y_interval=y_interval,
    # plot_type='streamline',
    plot_type="quiver",
    return_ax=False,
    vmax=1.4,
    quiver_spatial_res=0.1,
    quiver_scale=15
)
#%%
ocean_field.hindcast_data_source.animate_data(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    plot_type='streamline',
    output="test_hindcast_current_animation.mp4",
)
# it will be saved as file in the "generated_media" folder
#%% to get current data at a specific point
# For Hindcast Data
ocean_field.hindcast_data_source.update_casadi_dynamics(x_0)
hindcast_currents_at_point = ocean_field.get_ground_truth(x_0.to_spatio_temporal_point())
print("hindcast_currents_at_point: ", hindcast_currents_at_point)
#%% For Forecast Data
forecast_currents_at_point = ocean_field.get_forecast(x_0.to_spatio_temporal_point())
print("forecast_currents_at_point: ", forecast_currents_at_point)
#%% to access Data over an area
# Hindcast
hindcast_xarray = ocean_field.get_ground_truth_area(
    x_interval=x_interval, y_interval=y_interval, t_interval=t_interval
)
# Forecast
forecast_xarray = ocean_field.get_forecast_area(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    most_recent_fmrc_at_time=t_0 - datetime.timedelta(days=2),
)
#%% Calculate forecast RMSE from those
error_array = hindcast_xarray - forecast_xarray
print(
    "RMSE Error is: ",
    np.sqrt(error_array["water_u"] ** 2 + error_array["water_v"] ** 2).mean().item(),
)


#%% Analytical Ocean Current Example
#%% Highway current
true_current_source = {
    "field": "OceanCurrents",
    "source": "analytical",
    "source_settings": {
        "name": "FixedCurrentHighway",
        "boundary_buffers": [0.2, 0.2],
        "x_domain": [0, 10],
        "y_domain": [0, 10],
        "temporal_domain": [0, 10],
        "spatial_resolution": 0.1,
        "temporal_resolution": 1,
        "y_range_highway": [4, 6],
        "U_cur": 2,
    },
}
forecast_current_source = {
    "field": "OceanCurrents",
    "source": "analytical",
    "source_settings": {
        "name": "FixedCurrentHighway",
        "boundary_buffers": [0.2, 0.2],
        "x_domain": [0, 10],
        "y_domain": [0, 10],
        "temporal_domain": [0, 10],
        "spatial_resolution": 0.1,
        "temporal_resolution": 1,
        "y_range_highway": [4, 6],
        "U_cur": 8,
    },
}

#%% Periodic Double gyre
true_current_source = {
    "field": "OceanCurrents",
    "source": "analytical",
    "source_settings": {
        "name": "PeriodicDoubleGyre",
        "boundary_buffers": [0.2, 0.2],
        "x_domain": [-0.1, 2.1],
        "y_domain": [-0.1, 1.1],
        "temporal_domain": [-10, 1000],  # will be interpreted as POSIX timestamps
        "spatial_resolution": 0.05,
        "temporal_resolution": 10,
        "v_amplitude": 1,
        "epsilon_sep": 0.2,
        "period_time": 10,
    },
}
forecast_current_source = None
#%% Create the ocean Field
ocean_field = OceanCurrentField(
    hindcast_source_dict=true_current_source,
    forecast_source_dict=forecast_current_source,
    casadi_cache_dict=casadi_cache_dict,
    use_geographic_coordinate_system=False,
)
#%% visualize it at a specific time
ocean_field.plot_true_at_time_over_area(
    time=datetime.datetime.fromtimestamp(10, tz=datetime.timezone.utc),
    x_interval=[0, 10],
    y_interval=[0, 10],
)
