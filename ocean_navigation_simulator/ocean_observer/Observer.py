from __future__ import annotations

import datetime
from typing import Dict, Any, List, Union, Optional, Tuple

import numpy as np
import torch
import xarray
import xarray as xr

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.PlatformState import SpatioTemporalPoint
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentGP import OceanCurrentGP
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentModel import OceanCurrentModel
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentRunner import get_model


class Observer:
    """Class that represent the observer. It will receive observations and is then responsible to return predictions for
    the given areas.
    """

    def __init__(self, config: Dict[str, Any], device="cpu"):
        """Create the observer object
        Args:
            config: dictionary from the yaml file used to specify the parameters of the prediction model.
        """
        self.prediction_model = self.instantiate_model_from_dict(config.get("model"))
        print("Model: ", self.prediction_model)
        if "NN" in config.get("model", {}):
            NN_dict = config["model"]["NN"]
            self.model_error = NN_dict["model_error"]
            self.NN = get_model(NN_dict["type"], NN_dict["parameters"], device)
        else:
            self.NN = None

        self.forecast_data_source = None

    @staticmethod
    def instantiate_model_from_dict(source_dict: Dict[str, Any]) -> OceanCurrentModel:
        """Helper function to instantiate an OceanCurrentSource object from the dict
        Args:
            source_dict: dictionary that contains the model type and its parameters.

        Returns:
            The OceanCurrentModel generated.
        Raises:
             Value error if the model selected with source_dict is not supported (yet).
        """
        if "gaussian_process" in source_dict:
            return OceanCurrentGP(source_dict["gaussian_process"])

        raise ValueError(f"Selected model: {source_dict} in the OceanCurrentModel is not implemented.")

    @staticmethod
    def _convert_prediction_model_output(data: np.ndarray, reference_xr: xr,
                                         names_variables: Tuple[str, str]) -> xr:
        """Helper function to build a dataset given data in a numpy format and a xarray object as a reference for the
        dimensions and a tuple containing the name of the two variables that we include in the dataset.

        Args:
            data: array that contains the values we want to use as variables for the dataset we will generate
            reference_xr: xarray that is used to get the lon, lat and time axis
            names_variables: name of the two variables that will be added to the returned dataset

        Returns:
            The xr dataset generated with time lat and lon as the dimensions and names_variables as the variables
            computed using the 'data' array
        """
        data = data.reshape((len(reference_xr["lon"]), len(reference_xr["lat"]), len(reference_xr["time"]), 2))
        return xr.Dataset({names_variables[0]: (["time", "lat", "lon"], data[..., 0].swapaxes(0, 2)),
                           names_variables[1]: (["time", "lat", "lon"], data[..., 1].swapaxes(0, 2))},
                          coords={
                              "time": reference_xr['time'],
                              "lat": reference_xr['lat'],
                              "lon": reference_xr['lon']})

    def evaluate_neural_net(self, data):
        data_array = np.expand_dims(data.to_array().to_numpy(), 0)
        torch_data = torch.tensor(data_array[:, [0, 1, 2, 3, 4, 5]], dtype=torch.float)
        return self.NN(torch_data)[0].detach().numpy()
        # 1) Check the input size
        # if len(data["time"]) > n_steps:
        #     data = data.isel(time=slice(1, n_steps + 1))
        # 2) Pad with 0's
        # data_array = remove_borders_GP_predictions_lon_lat(data_array, radius_platform)
        # 3) evaluate NN

    def _get_predictions_from_GP(self, forecasts) -> xr:
        # Get all the points that we will query to the model as a 2D array. Coord field act like np.meshgrid but is
        # flattened
        coords = forecasts.stack(coord=["lon", "lat", "time"])["coord"].to_numpy()
        coords = np.array([*coords])
        # print("coords:", len(forecasts["lon"]), len(forecasts["lat"]))

        prediction_errors, prediction_std = self.prediction_model.get_predictions(coords)
        if prediction_std.shape != prediction_errors.shape:
            # print("Invalid version of SKlearn, the std values will not be correct")
            prediction_std = np.repeat(prediction_std[..., np.newaxis], 2, axis=-1)

        predictions_dataset = xr.merge(
            [self._convert_prediction_model_output(prediction_errors, forecasts, ("error_u", "error_v")),
             self._convert_prediction_model_output(prediction_std, forecasts, ("std_error_u", "std_error_v")),
             forecasts.rename(dict(water_u="initial_forecast_u", water_v="initial_forecast_v"))])

        predictions_dataset = predictions_dataset.assign({
            "water_u": lambda x: x.initial_forecast_u - x.error_u,
            "water_v": lambda x: x.initial_forecast_v - x.error_v})
        return predictions_dataset

    def get_data_over_area(self, x_interval: List[float], y_interval: List[float],
                           t_interval: List[Union[datetime.datetime, int]], spatial_resolution: Optional[float] = None,
                           temporal_resolution: Optional[float] = None) -> xarray:
        """Computes the xarray dataset that contains the prediction errors, the new forecasts (water_u & water_v), the
        old forecasts (renamed: initial_forecast_u & initial_forecast_v) and also std_error_u & std_error_v for the u
        and v directions respectively if the OceanCurrentModel output these values
        Args:
            x_interval: List of the lower and upper x area in the respective coordinate units [x_lower, x_upper]
            y_interval: List of the lower and upper y area in the respective coordinate units [y_lower, y_upper]
            t_interval: List of the lower and upper datetime requested [t_0, t_T] in datetime
            spatial_resolution: spatial resolution in the same units as x and y interval
            temporal_resolution: temporal resolution in seconds
        Returns:
            the computed xarray
        """
        forecasts = self.forecast_data_source.get_data_over_area(x_interval, y_interval, t_interval, spatial_resolution,
                                                                 temporal_resolution)

        return self._get_predictions_from_GP(forecasts)

    def get_data_around_platform(self, platform_position: SpatioTemporalPoint, radius_space: float,
                                 lags_in_second: Optional[int] = 43200, spatial_resolution: Optional[float] = None,
                                 temporal_resolution: Optional[float] = None) -> xarray:
        if spatial_resolution is None:
            spatial_resolution = 1 / 12
        if temporal_resolution is None:
            temporal_resolution = 3600

        # Compute the grid centered around the platform and interpolate the forecast based on that
        m = 1 / 20
        l1_lon = np.arange(platform_position.lon.deg, platform_position.lon.deg - radius_space - m, -spatial_resolution)
        l2_lon = np.arange(platform_position.lon.deg, platform_position.lon.deg + radius_space + m, spatial_resolution)
        l1_lat = np.arange(platform_position.lat.deg, platform_position.lat.deg - radius_space - m, -spatial_resolution)
        l2_lat = np.arange(platform_position.lat.deg, platform_position.lat.deg + radius_space + m, spatial_resolution)
        # margin
        m = np.array([-0.2, 0.2])
        lon, lat = np.concatenate((l1_lon[::-1], l2_lon[1:])), np.concatenate((l1_lat[::-1], l2_lat[1:]))
        time = np.array([platform_position.date_time + datetime.timedelta(seconds=s) for s in
                         range(0, lags_in_second, temporal_resolution)])
        m_t = np.array([-datetime.timedelta(hours=1), datetime.timedelta(hours=1)])
        time_in_np_format = np.array([np.datetime64(t) for t in time])
        model_coordinates = xr.Dataset(coords=dict(lon=lon, lat=lat, time=time_in_np_format))
        forecasts = self.forecast_data_source.get_data_over_area(lon[[0, -1]] + m, lat[[0, -1]] + m,
                                                                 time[[0, -1]] + m_t,
                                                                 spatial_resolution,
                                                                 temporal_resolution)
        forecasts_around_platform = forecasts.interp_like(model_coordinates)
        # todo: convert time dimension format
        improved_forecasts_around_platform = self._get_predictions_from_GP(forecasts_around_platform)
        assert (improved_forecasts_around_platform.lon == forecasts_around_platform.lon).all() and \
               (improved_forecasts_around_platform.lat == forecasts_around_platform.lat).all() and \
               (improved_forecasts_around_platform.time == forecasts_around_platform.time).all()
        return improved_forecasts_around_platform

    def get_data_at_point(self, lon: float, lat: float, time: datetime.datetime):
        coords = np.array([[lon, lat, time]])
        # print("coords:", len(forecasts["lon"]), len(forecasts["lat"]))

        prediction_errors, prediction_std = self.prediction_model.get_predictions(coords)
        return prediction_errors, prediction_std

    def fit(self) -> None:
        """Fit the inner prediction model using the observations recorded by the observer
        """
        self.prediction_model.fit()

    def reset(self) -> None:
        """Reset the observer and its prediction model
        """
        self.prediction_model.reset()

    def observe(self, arena_observation: ArenaObservation) -> None:
        """Add the error at the position of the arena_observation and also dave the forecast if it has not been done
        already
        Args:
            arena_observation: Observation where we compute the error and add to the observations
        """
        if self.forecast_data_source is None:
            self.forecast_data_source = arena_observation.forecast_data_source
            self.last_forecast_file = arena_observation.forecast_data_source.DataArray.encoding['source']

        observation_location = arena_observation.platform_state.to_spatio_temporal_point()
        measured_current_error = arena_observation.forecast_data_source.get_data_at_point(
            observation_location).subtract(arena_observation.true_current_at_state)

        # If the observer reads data from a new file --> Reset the observations
        if self.last_forecast_file != arena_observation.forecast_data_source.DataArray.encoding['source']:
            self.last_forecast_file = arena_observation.forecast_data_source.DataArray.encoding['source']
            self.reset()

        self.prediction_model.observe(observation_location, measured_current_error)

        # Todo: probably remove this. Added to stop if the forecast is missing
        if len(self.prediction_model.measurement_locations) > 24:
            UserWarning("Error: forecast file missing. Problem stopped")
            raise Exception("Error: forecast file missing. Problem stopped")

    # Forwarding functions as it replaces the forecast_data_source
    def check_for_most_recent_fmrc_dataframe(self, time: datetime.datetime) -> int:
        """Helper function to check update the self.OceanCurrent if a new forecast is available at
        the specified input time.
        Args:
          time: datetime object
        """
        return self.forecast_data_source.check_for_most_recent_fmrc_dataframe(time=time)
