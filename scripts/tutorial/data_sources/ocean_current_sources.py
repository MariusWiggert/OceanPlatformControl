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
# import yaml
# with open(f'config/arena/gulf_of_mexico_LongTermAverageSource.yaml') as f:
#     full_config = yaml.load(f, Loader=yaml.FullLoader)

# source_config_dict = full_config['ocean_dict']['forecast']
# forecast_dict = source_config_dict['source_settings']['forecast']
# average_dict = source_config_dict['source_settings']['average']
# for source_dict in [forecast_dict, average_dict]:
#     source_dict['casadi_cache_settings'] = casadi_cache_dict
#     source_dict['use_geographic_coordinate_system'] = True
# #%%
# from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource, ForecastFileSource, HindcastFileSource, get_datetime_from_np64

# # Step 1: Initialize both data_sources
# forecast_data_source = ForecastFileSource(forecast_dict)
# monthly_avg_data_source = HindcastFileSource(average_dict)
# #%% plot to see if it worked
# # forecast_data_source.plot_data_at_time_over_area(datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
# #                                                  [-84, -82], [20,25])
# # monthly_avg_data_source.plot_data_at_time_over_area(datetime.datetime(2021, 11, 30, 12, 0, tzinfo=datetime.timezone.utc),
# #                                                  [-84, -82], [20,25])
# #%%
# x_interval= [-84, -82]
# y_interval= [20,25]
# t_interval= [datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
#              datetime.datetime(2021, 11, 30, 12, 0, tzinfo=datetime.timezone.utc)]
# spatial_resolution = 0.1
# temporal_resolution= 3600*6
# #%%
# # todo: use keyword inputs for clarity!
# forecast_dataframe = forecast_data_source.get_data_over_area(x_interval, y_interval, t_interval,spatial_resolution, temporal_resolution)
# # 11-24 to 11-28
# # Now get end_forecast time
# end_forecast_time = get_datetime_from_np64(forecast_dataframe["time"].to_numpy()[-1])
# if end_forecast_time >= t_interval[1]:
#     print("forecast is enough")

# # easiest fix: run it with temp and spat resolution of the forecast...
# remaining_t_interval = [end_forecast_time, t_interval[1]]
# monthly_average_dataframe = monthly_avg_data_source.get_data_over_area(x_interval, y_interval,
#                                                                        remaining_t_interval,
#                                                                        spatial_resolution,
#                                                                        temporal_resolution)
# #%% Now cut out the relevant times:
# subset = monthly_average_dataframe.sel(
#             time=slice(remaining_t_interval[0], remaining_t_interval[1]))
# #%% now concat
# import xarray as xr
# full_frame = xr.concat([forecast_dataframe, subset], dim="time")
# # Works but right now it's multiple resolutions! (hourly vs monthly data...)
# #%% render it
# import datetime
# import matplotlib.animation as animation
# import numpy as np
# import xarray as xr
# from functools import partial

# # Step 1: get the data_subset for animation
# xarray = full_frame

# # Calculate min and max over the full tempo-spatial array
# # get rounded up vmax across the whole dataset (with ` decimals)
# xarray = xarray.assign(magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5)
# vmax = round(xarray['magnitude'].max().item() + 0.049, 1)
# vmin = 0


# # create global figure object where the animation happens
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12, 12))

# render_frame = partial(forecast_data_source.plot_xarray_for_animation, xarray=xarray, vmin=vmin, vmax=vmax,
#                                    reset_plot=True)
# # set time direction of the animation
# frames_vector = np.where(True, np.arange(xarray['time'].size), np.flip(np.arange(xarray['time'].size)))
# # create animation function object (it's not yet executed)
# ani = animation.FuncAnimation(fig, func=render_frame, frames=frames_vector, repeat=False)

# # render the animation with the keyword arguments
# forecast_data_source.render_animation(animation_object=ani, output="average.mp4", fps=10)
# #%%
# full_frame.sel(time=slice(remaining_t_interval[0] - datetime.timedelta(hours=10),
#                           remaining_t_interval[0] + datetime.timedelta(hours=10)))
# #%%
#     # Query as much forecast data as is possible
#     try:
#         forecast_dataframe = self.forecast_data_source.get_data_over_area(x_interval, y_interval, t_interval,
#                                                                           spatial_resolution, temporal_resolution)
#         end_forecast_time = get_datetime_from_np64(forecast_dataframe["time"].to_numpy()[-1])
#     except ValueError:
#         monthly_average_dataframe = self.monthly_avg_data_source.get_data_over_area(x_interval, y_interval,
#                                                                                     t_interval, spatial_resolution,
#                                                                                     temporal_resolution)
#         return monthly_average_dataframe

#     if end_forecast_time >= t_interval[1]:
#         return forecast_dataframe

#     remaining_t_interval = [end_forecast_time, t_interval[1]]  # may not work
#     monthly_average_dataframe = self.monthly_avg_data_source.get_data_over_area(x_interval, y_interval,
#                                                                                 remaining_t_interval,
#                                                                                 spatial_resolution,
#                                                                                 temporal_resolution)
#     return xr.concat([forecast_dataframe, monthly_average_dataframe], dim="time")


# #%%
# class LongTermAverageSource(OceanCurrentSource):
#     """"""

#     def __init__(self, source_config_dict: dict):
#         self.u_curr_func, self.v_curr_func = [None] * 2
#         self.forecast_data_source = ForecastFileSource(source_config_dict['source_settings']['forecast'])
#         self.monthly_avg_data_source = HindcastFileSource(
#             source_config_dict['source_settings']['average'])  # defaults currents to normal
#         self.source_config_dict = source_config_dict
#         # self.t_0 = source_config_dict['t0'] # not sure what to do here

#     def get_data_over_area(self, x_interval: List[float], y_interval: List[float],
#                            t_interval: List[Union[datetime.datetime, int]],
#                            spatial_resolution: Optional[float] = None,
#                            temporal_resolution: Optional[float] = None) -> xr:
#         # Query as much forecast data as is possible
#         try:
#             forecast_dataframe = self.forecast_data_source.get_data_over_area(x_interval, y_interval, t_interval,
#                                                                               spatial_resolution, temporal_resolution)
#             end_forecast_time = get_datetime_from_np64(forecast_dataframe["time"].to_numpy()[-1])
#         except ValueError:
#             monthly_average_dataframe = self.monthly_avg_data_source.get_data_over_area(x_interval, y_interval,
#                                                                                         t_interval, spatial_resolution,
#                                                                                         temporal_resolution)
#             return monthly_average_dataframe

#         if end_forecast_time >= t_interval[1]:
#             return forecast_dataframe

#         remaining_t_interval = [end_forecast_time, t_interval[1]]  # may not work
#         monthly_average_dataframe = self.monthly_avg_data_source.get_data_over_area(x_interval, y_interval,
#                                                                                     remaining_t_interval,
#                                                                                     spatial_resolution,
#                                                                                     temporal_resolution)
#         return xr.concat([forecast_dataframe, monthly_average_dataframe], dim="time")

#     def check_for_most_recent_fmrc_dataframe(self, time: datetime.datetime) -> int:
#         """Helper function to check update the self.OceanCurrent if a new forecast is available at
#         the specified input time.
#         Args:
#           time: datetime object
#         """
#         return self.forecast_data_source.check_for_most_recent_fmrc_dataframe(time)

#     # Not sure if I can just all this
#     def get_data_at_point(self, spatio_temporal_point: SpatioTemporalPoint) -> OceanCurrentVector:
#         """We overwrite it because we don't want that Forecast needs caching..."""
#         return self.forecast_data_source.get_data_at_point(spatio_temporal_point == spatio_temporal_point)







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
)
#%% animate the current evolution over time for the t_interval
ocean_field.hindcast_data_source.animate_data(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
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
