from __future__ import annotations

import datetime
from typing import Dict, Any, List, Union, Optional, Tuple

import numpy as np
import torch
import xarray
import xarray as xr

from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.PlatformState import SpatioTemporalPoint
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentGP import (
    OceanCurrentGP,
)
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentModel import (
    OceanCurrentModel,
)
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentRunner import (
    get_model,
)


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
            self.model_input = NN_dict["dimension_input"]
            self.NN = get_model(NN_dict["type"], NN_dict["parameters"], device)
        else:
            self.NN = None

        self.forecast_data_source = None
        self.last_forecast_file = None

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

        raise ValueError(
            f"Selected model: {source_dict} in the OceanCurrentModel is not implemented."
        )

    @staticmethod
    def get_grid_coordinates_around_platform(
        platform_position: SpatioTemporalPoint,
        radius_space: float,
        duration_tileset_in_seconds: Optional[int] = 43200,
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
        margin_space: Optional[float] = 1 / 20,
    ):
        if spatial_resolution is None:
            spatial_resolution = 1 / 12
        if temporal_resolution is None:
            temporal_resolution = 3600

        # Compute the grid centered around the platform and interpolate the forecast based on that

        l1_lon = np.arange(
            platform_position.lon.deg,
            platform_position.lon.deg - radius_space - margin_space,
            -spatial_resolution,
        )
        l2_lon = np.arange(
            platform_position.lon.deg,
            platform_position.lon.deg + radius_space + margin_space,
            spatial_resolution,
        )
        l1_lat = np.arange(
            platform_position.lat.deg,
            platform_position.lat.deg - radius_space - margin_space,
            -spatial_resolution,
        )
        l2_lat = np.arange(
            platform_position.lat.deg,
            platform_position.lat.deg + radius_space + margin_space,
            spatial_resolution,
        )
        lon, lat = np.concatenate((l1_lon[::-1], l2_lon[1:])), np.concatenate(
            (l1_lat[::-1], l2_lat[1:])
        )
        time = np.array(
            [
                platform_position.date_time + datetime.timedelta(seconds=s)
                for s in range(0, duration_tileset_in_seconds, temporal_resolution)
            ]
        )
        time_in_np_format = np.array([np.datetime64(t) for t in time])
        return lon, lat, time, time_in_np_format

    @staticmethod
    def _convert_prediction_model_output(
        data: np.ndarray, reference_xr: xr, names_variables: Tuple[str, str]
    ) -> xr:
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
        data = data.reshape(
            (
                len(reference_xr["lon"]),
                len(reference_xr["lat"]),
                len(reference_xr["time"]),
                2,
            )
        )
        return xr.Dataset(
            {
                names_variables[0]: (
                    ["time", "lat", "lon"],
                    data[..., 0].swapaxes(0, 2),
                ),
                names_variables[1]: (
                    ["time", "lat", "lon"],
                    data[..., 1].swapaxes(0, 2),
                ),
            },
            coords={
                "time": reference_xr["time"],
                "lat": reference_xr["lat"],
                "lon": reference_xr["lon"],
            },
        )

    def evaluate_neural_net(self, data):
        if self.NN is None:
            raise ValueError("The Neural Network is not specified correctly in the yaml file.")
        data_array = np.expand_dims(data.to_array().to_numpy(), 0)
        torch_data = torch.tensor(data_array, dtype=torch.float)
        # Only take the center of the tile to evaluate on the NN
        t, lon, lat = self.model_input
        margin_lon = (torch_data.shape[-2] - lon) // 2
        margin_lat = (torch_data.shape[-1] - lat) // 2
        t_xr, lon_xr, lat_xr = (
            data.time[:t],
            data.lon[margin_lon : margin_lon + lon],
            data.lat[margin_lat : margin_lat + lat],
        )
        torch_data = torch_data[
            :, :, :t, margin_lon : margin_lon + lon, margin_lat : margin_lat + lat
        ]
        output = self.NN(torch_data)[0].detach().numpy()
        if self.model_error:
            dict_output = dict(
                error_u=(["time", "lon", "lat"], output[0]),
                error_v=(["time", "lon", "lat"], output[1]),
            )
            dict_assign = dict(
                water_u=lambda x: x.initial_forecast_u - x.error_u,
                water_v=lambda x: x.initial_forecast_v - x.error_v,
            )
        else:
            dict_output = dict(
                water_u=(["time", "lon", "lat"], output[0]),
                water_v=(["time", "lon", "lat"], output[1]),
            )
            dict_assign = dict(
                error_u=lambda x: x.initial_forecast_u - x.water_u,
                error_v=lambda x: x.initial_forecast_v - x.water_v,
            )
        xr_output = xr.Dataset(
            data_vars=dict_output,
            coords=dict(time=t_xr, lat=lat_xr, lon=lon_xr),
        ).transpose("time", "lat", "lon")

        xr_output = xr.merge(
            [xr_output, data[["initial_forecast_u", "initial_forecast_v"]]], compat="override"
        )
        xr_output.assign(**dict_assign)

        return xr_output

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
            [
                self._convert_prediction_model_output(
                    prediction_errors, forecasts, ("error_u", "error_v")
                ),
                self._convert_prediction_model_output(
                    prediction_std, forecasts, ("std_error_u", "std_error_v")
                ),
                forecasts.rename(dict(water_u="initial_forecast_u", water_v="initial_forecast_v")),
            ]
        )

        predictions_dataset = predictions_dataset.assign(
            {
                "water_u": lambda x: x.initial_forecast_u - x.error_u,
                "water_v": lambda x: x.initial_forecast_v - x.error_v,
            }
        )
        return predictions_dataset

    def get_data_over_area(
        self,
        x_interval: List[float],
        y_interval: List[float],
        t_interval: List[Union[datetime.datetime, int]],
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
    ) -> xarray:
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
        forecasts = self.forecast_data_source.get_data_over_area(
            x_interval, y_interval, t_interval, spatial_resolution, temporal_resolution
        )

        return self._get_predictions_from_GP(forecasts)

    def get_data_around_platform(
        self,
        platform_position: SpatioTemporalPoint,
        radius_space: float,
        lags_in_second: Optional[int] = 43200,
        spatial_resolution: Optional[float] = None,
        temporal_resolution: Optional[float] = None,
    ) -> xarray:

        lon, lat, time, time_in_np_format = Observer.get_grid_coordinates_around_platform(
            platform_position=platform_position,
            radius_space=radius_space,
            duration_tileset_in_seconds=lags_in_second,
            spatial_resolution=spatial_resolution,
            temporal_resolution=temporal_resolution,
        )
        # margins
        margin_space = np.array([-0.2, 0.2])
        margin_time = np.array([-datetime.timedelta(hours=1), datetime.timedelta(hours=1)])

        model_coordinates = xr.Dataset(coords=dict(lon=lon, lat=lat, time=time_in_np_format))
        forecasts = self.forecast_data_source.get_data_over_area(
            lon[[0, -1]] + margin_space,
            lat[[0, -1]] + margin_space,
            time[[0, -1]] + margin_time,
            spatial_resolution,
            temporal_resolution,
        )
        forecasts_around_platform = forecasts.interp_like(model_coordinates)

        improved_forecasts_around_platform = self._get_predictions_from_GP(
            forecasts_around_platform
        )
        assert (
            (improved_forecasts_around_platform.lon == forecasts_around_platform.lon).all()
            and (improved_forecasts_around_platform.lat == forecasts_around_platform.lat).all()
            and (improved_forecasts_around_platform.time == forecasts_around_platform.time).all()
        )
        return improved_forecasts_around_platform

    def get_data_at_point(
        self, lon: float, lat: float, time: datetime.datetime
    ) -> [np.ndarray, np.ndarray]:
        coords = np.array([[lon, lat, time]])

        prediction_errors, prediction_std = self.prediction_model.get_predictions(coords)
        return prediction_errors, prediction_std

    def fit(self) -> None:
        """Fit the inner prediction model using the observations recorded by the observer"""
        self.prediction_model.fit()

    def reset(self) -> None:
        """Reset the observer and its prediction model"""
        self.prediction_model.reset()

    def observe(self, arena_observation: ArenaObservation) -> None:
        """Add the error at the position of the arena_observation and also dave the forecast if it has not been done
        already
        Args:
            arena_observation: Observation where we compute the error and add to the observations
        """
        if self.forecast_data_source is None:
            self.forecast_data_source = arena_observation.forecast_data_source
            self.last_forecast_file = arena_observation.forecast_data_source.DataArray.encoding[
                "source"
            ]

        observation_location = arena_observation.platform_state.to_spatio_temporal_point()
        measured_current_error = arena_observation.forecast_data_source.get_data_at_point(
            observation_location
        ).subtract(arena_observation.true_current_at_state)

        # If the observer reads data from a new file --> Reset the observations
        if (
            self.last_forecast_file
            != arena_observation.forecast_data_source.DataArray.encoding["source"]
        ):
            self.last_forecast_file = arena_observation.forecast_data_source.DataArray.encoding[
                "source"
            ]
            self.reset()

        self.prediction_model.observe(observation_location, measured_current_error)

        if len(self.prediction_model.measurement_locations) > 24:
            raise UserWarning(
                f"Error: forecast file missing. Problem stopped: {arena_observation.platform_state.to_spatio_temporal_point().date_time}"
            )

    # Forwarding functions as it replaces the forecast_data_source
    def check_for_most_recent_fmrc_dataframe(self, time: datetime.datetime) -> int:
        """Helper function to check update the self.OceanCurrent if a new forecast is available at
        the specified input time.
        Args:
          time: datetime object
        """
        return self.forecast_data_source.check_for_most_recent_fmrc_dataframe(time=time)
