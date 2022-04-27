import datetime
from typing import List

import numpy as np
import pandas as pd

from ocean_navigation_simulator.env.Arena import Arena
from ocean_navigation_simulator.env.PlatformState import SpatialPoint, PlatformState
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.utils.units import Distance
from scripts.experiments.class_gp import OceanCurrentGP

_DT_LAT, _DT_LONG = units.Distance(deg=0.5), units.Distance(deg=0.5)
#TODO: Change that such that the value is a function depending on the source of the hindcast
_LAT_DEG_PER_CELL_HINDCAST, _LONG_DEG_PER_CELL_HINDCAST = Distance(deg=0.04), Distance(deg=0.04)
_T_HORIZON = datetime.timedelta(days=3)

class Observer:
    def __init__(self, prediction_model: OceanCurrentGP, arena: Arena):
        self.model = prediction_model
        self.arena = arena

    def __get_area(self, platform_state: PlatformState, forecast: bool):
        fn = self.arena.ocean_field.get_forecast_area if forecast else self.arena.ocean_field.get_ground_truth_area
        return fn([(platform_state.lon - _DT_LONG).deg, (platform_state.lon + _DT_LONG).deg],
                  [(platform_state.lat - _DT_LAT).deg, (platform_state.lat + _DT_LAT).deg],
                  [platform_state.date_time, platform_state.date_time + _T_HORIZON])

    def __get_forecasts_area(self, platform_state: PlatformState):
        return self.__get_area(platform_state, True)

    def __get_ground_truth_area(self, platform_state: PlatformState):
        return self.__get_area(platform_state, False)

    def __get_closest_point_platform(self, platform_state: PlatformState):
        lon, lat, t = platform_state.lon, platform_state.lat, platform_state.date_time
        self.arena.ocean_field.get_ground_truth_area(
            [(lon-_LONG_DEG_PER_CELL_HINDCAST/2).deg, (lon-_LONG_DEG_PER_CELL_HINDCAST/2).deg],
            [(lat-_LAT_DEG_PER_CELL_HINDCAST/2).deg, (lat+_LAT_DEG_PER_CELL_HINDCAST/2).deg],
            [t-datetime.timedelta(days=1), t])

    def fit_and_evaluate(self, platform_state: PlatformState):
        forecast_area = self.__get_forecasts_area(platform_state)
        # hindcast_area = self.__get_forecasts_area(platform_state)
        locations = np.array(
            np.meshgrid(forecast_area["lon"], forecast_area["lat"], pd.to_datetime(forecast_area["time"]))).T.reshape(
            (-1, 3))
        locations[:, 2] = pd.to_datetime(locations[:, 2])
        # self.model.observe(platform_state.lon, platform_state.lat,platform_state.date_time,)
        return self.model.query_locations(locations)


    def observe(self, platform_state: PlatformState, difference_forecast_gt: OceanCurrentVector):
        self.model.observe(platform_state.lon, platform_state.lat,platform_state.date_time, difference_forecast_gt)