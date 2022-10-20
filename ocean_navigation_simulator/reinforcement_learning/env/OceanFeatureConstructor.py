import datetime
from typing import Tuple, Union

import gym
import numpy as np
import yaml

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.ocean_observer.Observer import Observer


class OceanFeatureConstructor:
    """
    Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
    and then convert to a numpy array that the RL model can use.
    """

    def __init__(
        self, config: dict, forecast_planner: HJReach2DPlanner, hindcast_planner: HJReach2DPlanner
    ):
        self.forecast_planner = forecast_planner
        self.hindcast_planner = hindcast_planner
        self.config = config

        # Step 1: Initialize Measurements
        if self.config["measurements"] > 0:
            self.measurements = np.zeros(shape=(0, 4))

        # Step 2: Initialize Observer
        if len(self.config["local_map"]["features"]["observer"]["variables"]) > 0:
            with open("config/reinforcement_learning/gaussian_process.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.observer = Observer(config["observer"])

    @staticmethod
    def get_observation_space(config):
        features = []

        # # Step 1: Raw Measurements
        if config["measurements"]:
            features += [
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(config["measurements"], 4),
                    dtype=np.float32,
                )
            ]

        # Step 2: Local Map
        if config["local_map"]:
            features += [
                OceanFeatureConstructor.get_map_space_from_config(config=config["local_map"])
            ]

        # Step 3: Global Map
        if config["global_map"]:
            features += [
                OceanFeatureConstructor.get_map_space_from_config(config=config["global_map"])
            ]

        # Step 4: Meta Information
        if config["meta"]:
            meta_features = len(config["meta"])
            features += [
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(meta_features,),
                    dtype=np.float32,
                )
            ]

        if config["flatten"]:
            return gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(sum([int(np.prod(f.shape)) for f in features]),),
                dtype=np.float32,
            )
        elif len(features) > 1:
            return gym.spaces.Tuple([f for f in features])
        else:
            return features[0]

    @staticmethod
    def get_map_space_from_config(config):
        map_features = (
            config["features"]["ttr_forecast"]
            + config["features"]["ttr_hindcast"]
            + len(config["features"]["observer"]["variables"])
            * len(config["features"]["observer"]["time"])
            + 2 * len(config["features"]["currents_hindcast"])
            + 2 * len(config["features"]["currents_forecast"])
            + 2 * len(config["features"]["true_error"])
        )
        if config["flatten"]:
            shape = (config["xy_width_points"] ** 2 * map_features,)
        elif map_features > 1:
            shape = (config["xy_width_points"], config["xy_width_points"], map_features)
        else:
            shape = (config["xy_width_points"], config["xy_width_points"])
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=shape,
            dtype=np.float32,
        )

    def get_features_from_state(
        self, fc_obs: ArenaObservation, hc_obs: ArenaObservation, problem: NavigationProblem
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        features = ()

        # # Step 1: Raw Measurements
        if self.config["measurements"]:
            features += (
                self.get_measurements_from_state(
                    fc_obs, hc_obs, problem, self.config["measurements"]
                ),
            )

        # Step 2: Local Map
        if self.config["local_map"]:
            features += (
                self.get_map_from_state(fc_obs, hc_obs, problem, self.config["local_map"]),
            )

        # Step 3: Global Map
        if self.config["global_map"]:
            features += (
                self.get_map_from_state(fc_obs, hc_obs, problem, self.config["global_map"]),
            )

        # Step 4: Meta Information
        if self.config["meta"]:
            features += (self.get_meta_from_state(fc_obs, hc_obs, problem, self.config["meta"]),)

        if self.config["flatten"]:
            return np.concatenate(tuple(f.flatten() for f in features))
        elif len(features) > 1:
            return features
        else:
            return features[0]

    def get_measurements_from_state(self, fc_obs, hc_obs, problem, config) -> np.ndarray:
        self.measurements = np.append(
            self.measurements[1 - config :],
            np.array(
                [
                    [
                        fc_obs.platform_state.lon.deg,
                        fc_obs.platform_state.lat.deg,
                        fc_obs.true_current_at_state.u,
                        fc_obs.true_current_at_state.v,
                    ]
                ]
            ),
            axis=0,
        )
        measurements = (
            self.measurements
            - np.repeat(np.array([[1, 1, 0, 0]]), repeats=self.measurements.shape[0], axis=0)
            * self.measurements[-1]
        )
        repeats = np.repeat(
            measurements[-1:], repeats=(config - self.measurements.shape[0]), axis=0
        )
        return np.append(measurements, repeats, axis=0)

    def get_map_from_state(self, fc_obs, hc_obs, problem, config) -> np.ndarray:
        x_interval = [
            fc_obs.platform_state.lon.deg - config["xy_width_degree"] / 2,
            fc_obs.platform_state.lon.deg + config["xy_width_degree"] / 2,
        ]
        y_interval = [
            fc_obs.platform_state.lat.deg - config["xy_width_degree"] / 2,
            fc_obs.platform_state.lat.deg + config["xy_width_degree"] / 2,
        ]
        map = np.zeros((config["xy_width_points"], config["xy_width_points"], 0), dtype="float32")

        # Step 1: TTR Map
        if config["features"]["ttr_forecast"]:
            ttr_forecast = self.forecast_planner.interpolate_value_function_in_hours(
                fc_obs, width_deg=config["xy_width_degree"], width=config["xy_width_points"]
            )
            map = np.concatenate(
                (map, np.expand_dims(ttr_forecast, axis=2)), axis=2, dtype="float32"
            )
        if config["features"]["ttr_hindcast"]:
            ttr_hindcast = self.hindcast_planner.interpolate_value_function_in_hours(
                hc_obs, width_deg=config["xy_width_degree"], width=config["xy_width_points"]
            )
            map = np.concatenate(
                (map, np.expand_dims(ttr_hindcast, axis=2)), axis=2, dtype="float32"
            )

        # Step 2: GP Observer
        if len(config["features"]["observer"]["variables"]) > 0:
            t_interval = [
                hc_obs.platform_state.date_time + datetime.timedelta(hours=h)
                for h in config["features"]["observer"]["time"]
            ]
            self.observer.observe(fc_obs)
            self.observer.fit()
            gp_map = (
                self.observer.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                )[config["features"]["observer"]["variables"]]
                .to_array()
                .to_numpy()
                .astype("float32")
            )
            gp_map = np.moveaxis(gp_map, [0, 1, 2, 3], [2, 3, 0, 1]).reshape(
                (config["xy_width_points"], config["xy_width_points"], -1)
            )
            map = np.concatenate((map, gp_map), axis=2, dtype="float32")

        # Step 3: Hindcast Currents
        if len(config["features"]["currents_hindcast"]) > 0:
            t_interval = [
                hc_obs.platform_state.date_time + datetime.timedelta(hours=h)
                for h in config["features"]["currents_hindcast"]
            ]
            current_map = (
                hc_obs.forecast_data_source.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                )["water_u", "water_v"]
                .to_array()
                .to_numpy()
                .astype("float32")
            )
            current_map = np.moveaxis(current_map, [0, 1, 2, 3], [2, 3, 0, 1]).reshape(
                (config["xy_width_points"], config["xy_width_points"], -1)
            )
            map = np.concatenate((map, current_map.squeeze()), axis=2, dtype="float32")

        # Step 4: Forecast Currents
        if len(config["features"]["currents_forecast"]) > 0:
            t_interval = [
                fc_obs.platform_state.date_time + datetime.timedelta(hours=h)
                for h in config["features"]["currents_forecast"]
            ]
            current_map = (
                fc_obs.forecast_data_source.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                )["water_u", "water_v"]
                .to_array()
                .to_numpy()
                .astype("float32")
            )
            current_map = np.moveaxis(current_map, [0, 1, 2, 3], [2, 3, 0, 1]).reshape(
                (config["xy_width_points"], config["xy_width_points"], -1)
            )
            map = np.concatenate((map, current_map.squeeze()), axis=2, dtype="float32")

        # Step 5: True Error
        if len(config["features"]["true_error"]) > 0:
            t_interval = [
                fc_obs.platform_state.date_time + datetime.timedelta(hours=h)
                for h in config["features"]["true_error"]
            ]
            current_fc = (
                fc_obs.forecast_data_source.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                )
                .to_array()
                .to_numpy()
                .astype("float32")
            )
            current_fc = np.moveaxis(current_fc, [0, 1, 2, 3], [2, 3, 0, 1]).reshape(
                (config["xy_width_points"], config["xy_width_points"], -1)
            )
            current_hc = (
                hc_obs.forecast_data_source.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                )
                .to_array()
                .to_numpy()
                .astype("float32")
            )
            current_hc = np.moveaxis(current_hc, [0, 1, 2, 3], [2, 3, 0, 1]).reshape(
                (config["xy_width_points"], config["xy_width_points"], -1)
            )
            error_map = current_fc - current_hc
            map = np.concatenate((map, error_map.squeeze()), axis=2, dtype="float32")

        return map.flatten() if config["flatten"] else map.squeeze()

    def get_meta_from_state(self, fc_obs, hc_obs, problem, config) -> np.ndarray:
        features = []

        if "lon" in config:
            features += [fc_obs.platform_state.lon.deg]
        if "lat" in config:
            features += [fc_obs.platform_state.lon.deg]
        if "time" in config:
            features += [fc_obs.platform_state.date_time.timestamp()]
        if "target_distance" in config:
            features += [problem.distance(fc_obs.platform_state)]
        if "target_direction" in config:
            features += [problem.angle(fc_obs.platform_state)]
        if "episode_time_in_h" in config:
            features += [problem.passed_seconds(fc_obs.platform_state) / 3600]
        if "ttr_center" in config:
            features += [
                self.forecast_planner.interpolate_value_function_in_hours(fc_obs).squeeze()
            ]

        return np.array(features)
