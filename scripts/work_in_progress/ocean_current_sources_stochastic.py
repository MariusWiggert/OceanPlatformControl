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
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import (
    GroundTruthFromNoise,
    HindcastFileSource, HindcastOpendapSource
)
import xarray as xr
#%%
ensemble1 = xr.open_dataset("data/mseas_ensemble/pe_ens_001.nc")
#%%
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
#%% Step 1: Fix the plotting issues why is it 200 degree things?
# The HYCOM data sources are in negative degrees... Copernicus is -180 to + 180 from opendap...
# It's HYCOM global hindcast files that are 0-360, the rest is fine!
# => for now let's just not use that! But use Copernicus Global!
# => let's make a check in there and normalize to -180, + 180==> DONE!
#%% Step 2: Implement the sources we actually need (with ensemble and without ensemble)

# Step 3: So that it can be loaded from a dict
# Ideally make it general so that if
# Step 4: Implement stochastic version locally
# Step 5: Start experiment on the cloud, compare stochastic with deterministic version! (Calculate also error distribution)


# define intervals
lon_interval = [-140, -135]
lat_interval = [20, 25]
t_interval = [datetime.datetime(2022, 10, 1, 12, 30, 0), datetime.datetime(2022, 10, 8, 12, 30, 0)]
target_folder = "data/hycom_hindcast_gen_noise_test/" # this is Region 1 data

# TODO: Implement animation function that shows hindcast, noise, and both together in a video over time!
# TODO: ultimately we want an ocean field to be as easy to instantiate as the others, directly from one dict, build the constructors to do that.
# #%% download files if not there yet
# ArenaFactory.download_required_files(
#                 archive_source='hycom',
#                 archive_type='hindcast',
#                 region='Region 1',
#                 download_folder=target_folder,
#                 t_interval=t_interval
#             )
#%% Getting the GT data Source by adding Generative Noise
source_dict = {
    "field": "OceanCurrents",
    "source": "hindcast_files",
    'use_geographic_coordinate_system': True,
    "source_settings": {
        "folder": target_folder},
}
hindcast_data = HindcastFileSource(source_dict)
#% Initialize the GT from Noise field!
gt = GroundTruthFromNoise(
    seed=100, # this needs to be an integer
    params_path="ocean_navigation_simulator/generative_error_model/models/"
                + "tuned_2d_forecast_variogram_area1_[5.0, 1.0]_False_True.npy",
    hindcast_data_source=hindcast_data,
)
#%%
gt.plot_noise_at_time_over_area(time=t_interval[0], x_interval=lon_interval, y_interval=lat_interval)
#%% plot comparison
# without noise
hindcast_data.plot_data_at_time_over_area(time=t_interval[0], x_interval=lon_interval, y_interval=lat_interval)
#%% # with Noise
gt.plot_data_at_time_over_area(time=t_interval[0], x_interval=lon_interval, y_interval=lat_interval)








#%%
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
t_interval = [t_0, t_0 + datetime.timedelta(days=4)]
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
)
#%% animate the current evolution over time for the t_interval
ocean_field.hindcast_data_source.animate_data(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    output="test_copernicus_current_animation.mp4",
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
