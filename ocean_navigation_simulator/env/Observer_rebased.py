from __future__ import annotations

import datetime
from typing import Dict, Any, List, Union, Optional, Tuple

import numpy as np
import xarray
import xarray as xr

from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.models.OceanCurrentGP_rebased import OceanCurrentGP
from ocean_navigation_simulator.env.models.OceanCurrentModel_rebased import OceanCurrentModel
from ocean_navigation_simulator.utils.corrected_forecast_utils import get_error_ocean_current_vector, \
    get_improved_forecast


def _convert_prediction_model_output(data: np.ndarray, reference_array: xr,
                                     names_currents: Tuple[str, str]) -> xr:
    data = data.reshape(
        (len(reference_array["lon"]), len(reference_array["lat"]), len(reference_array["time"]), 2))
    return xr.Dataset({names_currents[0]: (["time", "lat", "lon"], data[..., 0].swapaxes(0, 2)),
                       names_currents[1]: (["time", "lat", "lon"], data[..., 0].swapaxes(0, 2))},
                      coords={
                          "time": reference_array['time'],
                          "lat": reference_array['lat'],
                          "lon": reference_array['lon']})


class Observer:
    def __init__(self, config: Dict[str, Any]):
        self.prediction_model = self.instantiate_model_from_dict(config.get("model"))
        print("Model: ", self.prediction_model)
        self.forecast_data_source = None

    @staticmethod
    def instantiate_model_from_dict(source_dict: Dict[str, Any]) -> OceanCurrentModel:
        """Helper function to instantiate an OceanCurrentSource object from the dict."""
        if "gaussian_process" in source_dict:
            return OceanCurrentGP(source_dict["gaussian_process"])

        ValueError(f"Selected model: {source_dict} in the OceanCurrentModel is not implemented.")

    # Todo: change the interface
    def get_data_over_area(self, x_interval: List[float], y_interval: List[float],
                           t_interval: List[Union[datetime.datetime, int]], spatial_resolution: Optional[float] = None,
                           temporal_resolution: Optional[float] = None) -> xarray:

        forecasts = self.forecast_data_source.get_data_over_area(x_interval, y_interval, t_interval, spatial_resolution,
                                                                 temporal_resolution)
        # Get all the points that we will query to the model as a 2D array. Coord field act like np.meshgrid but is
        # flattened
        coords = forecasts.stack(coord=["lon", "lat", "time"])["coord"].to_numpy()
        coords = np.array([*coords])

        prediction_errors, prediction_std = self.prediction_model.get_predictions(coords)
        predictions_dataset = xr.merge(
            [_convert_prediction_model_output(prediction_errors, forecasts, ("mean_error_u", "mean_error_v")),
             _convert_prediction_model_output(prediction_std, forecasts, ("std_error_u", "std_error_v")),
             forecasts.rename(dict(water_u="initial_forecast_u", water_v="initial_forecast_v"))])

        predictions_dataset = predictions_dataset.assign(
            water_u=lambda x: get_improved_forecast(x.initial_forecast_u, x.mean_error_u))
        predictions_dataset = predictions_dataset.assign(
            water_v=lambda x: get_improved_forecast(x.initial_forecast_v, x.mean_error_v))
        return predictions_dataset

    def fit(self) -> None:
        self.prediction_model.fit()

    def reset(self) -> None:
        self.prediction_model.reset()

    def observe(self, arena_observation: ArenaObservation) -> None:
        if self.forecast_data_source is None:
            self.forecast_data_source = arena_observation.forecast_data_source

        position_forecast = arena_observation.platform_state.to_spatio_temporal_point()
        error = get_error_ocean_current_vector(arena_observation.forecast_data_source.get_data_at_point(
            position_forecast), arena_observation.true_current_at_state)
        self.prediction_model.observe(position_forecast, error)
