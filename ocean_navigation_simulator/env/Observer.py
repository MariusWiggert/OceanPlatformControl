import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ocean_navigation_simulator.env.Arena import Arena
from ocean_navigation_simulator.env.PlatformState import SpatialPoint, PlatformState, SpatioTemporalPoint
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.utils.units import Distance
from scripts.experiments.class_gp import OceanCurrentGP

_DT_LAT, _DT_LONG = units.Distance(deg=0.5), units.Distance(deg=0.5)
# TODO: Change that such that the value is a function depending on the source of the hindcast
_LAT_DEG_PER_CELL_HINDCAST, _LONG_DEG_PER_CELL_HINDCAST = Distance(deg=0.04), Distance(deg=0.04)
_T_HORIZON = datetime.timedelta(hours=1)


class Observer:
    def __init__(self, prediction_model: OceanCurrentGP, arena: Arena):
        self.model = prediction_model
        self.arena = arena
        self.temporal_resolution = arena.ocean_field.forecast_data_source.temporal_resolution

    def __get_area(self, platform_state: PlatformState, forecast: bool, x_y_intervals=None) -> xr:
        fn = self.arena.ocean_field.get_forecast_area if forecast else self.arena.ocean_field.get_ground_truth_area
        if x_y_intervals is None:
            x_y_intervals = ([(platform_state.lon - _DT_LONG).deg, (platform_state.lon + _DT_LONG).deg],
                             [(platform_state.lat - _DT_LAT).deg, (platform_state.lat + _DT_LAT).deg])

        return fn(*x_y_intervals, [platform_state.date_time + datetime.timedelta(seconds=self.temporal_resolution),
                                   platform_state.date_time + _T_HORIZON])

    def __get_forecasts_area(self, platform_state: PlatformState, x_y_intervals=None) -> xr:
        return self.__get_area(platform_state, True, x_y_intervals=x_y_intervals)

    def __get_ground_truth_area(self, platform_state: PlatformState, x_y_intervals=None) -> xr:
        return self.__get_area(platform_state, False, x_y_intervals=x_y_intervals)

    # Probably useless
    '''
    def __get_closest_point_platform(self, platform_state: PlatformState, only_around_platform=True):
        lon, lat, t = platform_state.lon, platform_state.lat, platform_state.date_time
        if only_around_platform:
            return self.arena.ocean_field.get_ground_truth_area(
                [(lon - _LONG_DEG_PER_CELL_HINDCAST / 2).deg, (lon - _LONG_DEG_PER_CELL_HINDCAST / 2).deg],
                [(lat - _LAT_DEG_PER_CELL_HINDCAST / 2).deg, (lat + _LAT_DEG_PER_CELL_HINDCAST / 2).deg],
                [t - datetime.timedelta(days=1), t])
        # we return the predictions for the whole grid (Used in simulation mode)
        else:
            return self.arena.ocean_field.get_ground_truth_area(
                [self.arena.seaweed_field.forecast_data_source]
            )
    '''

    def evaluate(self,point: SpatioTemporalPoint) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.query_locations(np.array([[point.lat.deg, point.lon.deg, point.date_time]]))

    def fit_and_evaluate(self, platform_state: PlatformState, x_y_intervals=None, delta=datetime.timedelta(seconds=0)) -> Tuple[xr.DataArray, xr.DataArray]:
        # 1) Fit the model
        self.model.fitting_GP()

        # 2) Query the whole grid around the platform
        area = self.__get_forecasts_area(platform_state, x_y_intervals=x_y_intervals)
        coords = np.array(np.meshgrid(area["lon"], area["lat"], pd.to_datetime(area["time"]))).T
        locations = coords.reshape((-1, 3))
        locations[:, 2] = pd.to_datetime(locations[:, 2]) + delta
        mean, std = self.model.query_locations(locations)
        reshape_dims = (*coords.shape[:-1], 2)
        mean, std = mean.reshape(reshape_dims), std.reshape(reshape_dims)
        mean_xr = xr.DataArray(data=mean, dims=["time", "lon", "lat", "u_v"],
                               coords={"time": area["time"], "lon": area["lon"], "lat": area["lat"], "u_v": ["u", "v"]})
        std_xr = xr.DataArray(data=std, dims=["time", "lon", "lat", "u_v"],
                              coords={"time": area["time"], "lon": area["lon"], "lat": area["lat"], "u_v": ["u", "v"]})
        return mean_xr, std_xr

    def observe(self, platform_state: PlatformState, difference_forecast_gt: OceanCurrentVector):
        self.model.observe(platform_state.lat.deg, platform_state.lon.deg, platform_state.date_time,
                           difference_forecast_gt)
