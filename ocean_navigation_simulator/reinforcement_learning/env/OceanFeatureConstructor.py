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
        if (
            self.config["local_map"]
            and len(self.config["local_map"]["features"]["observer_variables"]) > 0
        ):
            with open("config/reinforcement_learning/gaussian_process.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.observer = Observer(config["observer"])

    @staticmethod
    def get_observation_space(config):
        features = {}

        # # Step 1: Raw Measurements
        if config["measurements"]:
            features["measurement"] = gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(config["measurements"], 4),
                dtype=np.float32,
            )

        # Step 2: Local Map
        if config["local_map"]:
            features["local_map"] = OceanFeatureConstructor.get_map_space_from_config(
                config=config["local_map"]
            )

        # Step 3: Global Map
        if config["global_map"]:
            features["global_map"] = OceanFeatureConstructor.get_map_space_from_config(
                config=config["global_map"]
            )

        # Step 4: Meta Information
        if config["meta"]:
            meta_features = len(config["meta"])
            features["meta"] = gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(meta_features,),
                dtype=np.float32,
            )

        if config["flatten"]:
            return gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(sum([int(np.prod(f.shape)) for f in features.values()]),),
                dtype=np.float32,
            )
        else:
            return gym.spaces.Dict(spaces=features)

    @staticmethod
    def get_map_space_from_config(config):
        map_features = len(config["hours"]) * (
            config["features"]["ttr_forecast"]
            + config["features"]["ttr_hindcast"]
            + len(config["features"]["observer_variables"])
            + 2 * config["features"]["currents_hindcast"]
            + 2 * config["features"]["currents_forecast"]
            + 2 * config["features"]["true_error"]
        )
        if "xy_width_degree" in config and "xy_width_points" in config:
            if config["flatten"]:
                shape = (map_features * config["xy_width_points"] ** 2,)
            else:
                shape = (map_features, config["xy_width_points"], config["xy_width_points"])
        else:
            spatial_points = sum(config["embedding_n"])
            if config["flatten"]:
                shape = (map_features * spatial_points,)
            else:
                shape = (map_features, spatial_points)
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=shape,
            dtype=np.float32,
        )

    def get_features_from_state(
        self, obs: ArenaObservation, problem: NavigationProblem
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        features = {}

        # # Step 1: Raw Measurements
        if self.config["measurements"]:
            features["measurements"] = (
                self.get_measurements_from_state(obs, problem, self.config["measurements"]),
            )

        # Step 2: Local Map
        if self.config["local_map"]:
            features["local_map"] = self.get_map_from_state(obs, problem, self.config["local_map"])

        # Step 3: Global Map
        if self.config["global_map"]:
            features["global_map"] = self.get_map_from_state(
                obs, problem, self.config["global_map"]
            )

        # Step 4: Meta Information
        if self.config["meta"]:
            features["meta"] = self.get_meta_from_state(obs, problem, self.config["meta"])

        if self.config["flatten"]:
            return np.concatenate(tuple(f.flatten() for f in features.values()))
        else:
            return features

    def get_measurements_from_state(self, obs, problem, config) -> np.ndarray:
        self.measurements = np.append(
            self.measurements[1 - config :],
            np.array(
                [
                    [
                        obs.platform_state.lon.deg,
                        obs.platform_state.lat.deg,
                        obs.true_current_at_state.u,
                        obs.true_current_at_state.v,
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

    def get_map_from_state(self, obs, problem, config) -> np.ndarray:
        if "xy_width_degree" in config and "xy_width_points" in config:
            x_interval = [
                obs.platform_state.lon.deg - config["xy_width_degree"] / 2,
                obs.platform_state.lon.deg + config["xy_width_degree"] / 2,
            ]
            y_interval = [
                obs.platform_state.lat.deg - config["xy_width_degree"] / 2,
                obs.platform_state.lat.deg + config["xy_width_degree"] / 2,
            ]
        t_interval = [
            obs.platform_state.date_time + datetime.timedelta(hours=h) for h in config["hours"]
        ]
        maps = []

        # Step 1: TTR FC
        if config["features"]["ttr_forecast"]:
            if "xy_width_degree" in config and "xy_width_points" in config:
                ttr = self.forecast_planner.interpolate_value_function_in_hours(
                    point=obs.platform_state.to_spatio_temporal_point(),
                    width_deg=config["xy_width_degree"],
                    width=config["xy_width_points"],
                    h_grid=config["hours"],
                    allow_spacial_extrapolation=True,
                    allow_temporal_extrapolation=True,
                )
                # Interpolator: x, y, t
                # Need: 1, t, x, y
                if len(ttr.shape) > 2:
                    ttr = np.moveaxis(ttr, [0, 1, 2], [1, 2, 0])
                else:
                    ttr = np.expand_dims(ttr, axis=0)
                ttr = np.expand_dims(ttr, axis=0)
            else:
                ttr = self.forecast_planner.interpolate_value_function_in_hours(
                    point=obs.platform_state.to_spatio_temporal_point(),
                    embedding_n=config["embedding_n"],
                    embedding_radius=config["embedding_radius"],
                    h_grid=config["hours"],
                    allow_spacial_extrapolation=True,
                    allow_temporal_extrapolation=True,
                )
            maps.append(ttr)
        # Step 2: TTR HC
        if config["features"]["ttr_hindcast"]:
            if "xy_width_degree" in config and "xy_width_points" in config:
                ttr = self.hindcast_planner.interpolate_value_function_in_hours(
                    point=obs.platform_state.to_spatio_temporal_point(),
                    width_deg=config["xy_width_degree"],
                    width=config["xy_width_points"],
                    h_grid=config["hours"],
                    allow_spacial_extrapolation=True,
                    allow_temporal_extrapolation=True,
                )
                # Interpolator: x, y, t
                # Need: 1, t, x, y
                if len(ttr.shape) > 2:
                    ttr = np.moveaxis(ttr, [0, 1, 2], [1, 2, 0])
                else:
                    ttr = np.expand_dims(ttr, axis=0)
                ttr = np.expand_dims(ttr, axis=0)
            else:
                ttr = self.hindcast_planner.interpolate_value_function_in_hours(
                    point=obs.platform_state.to_spatio_temporal_point(),
                    embedding_n=config["embedding_n"],
                    embedding_radius=config["embedding_radius"],
                    h_grid=config["hours"],
                    allow_spacial_extrapolation=True,
                    allow_temporal_extrapolation=True,
                )
            maps.append(ttr)
        # Step 3: GP Observer
        if len(config["features"]["observer_variables"]) > 0:
            self.observer.observe(obs)
            self.observer.fit()
            variables = config["features"]["observer_variables"]
            if (
                "mag" in config["features"]["observer_variables"]
                and "dir" in config["features"]["observer_variables"]
            ):
                variables.remove("mag")
                variables.remove("dir")
                variables = ["error_u", "error_v"] + variables

            if "xy_width_degree" in config and "xy_width_points" in config:
                # Observer: var (0), t (1), x (2), y (3)
                gp_map = (
                    self.observer.get_data_over_area(
                        x_interval=x_interval,
                        y_interval=y_interval,
                        t_interval=[t_interval[0], t_interval[-1]],
                        throw_exceptions=False,
                    )
                    .interp(
                        time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                        lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                        lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                        method="linear",
                        kwargs={
                            "fill_value": "extrapolate",
                        },
                    )[variables]
                    .to_array()
                    .to_numpy()
                )
            else:
                point = obs.platform_state.to_spatio_temporal_point()
                gp_points = []
                for i in range(len(config["embedding_n"])):
                    for angle in np.linspace(0, 2 * np.pi, config["embedding_n"][i], endpoint=False):
                        out_x = point.lon.deg + np.sin(angle) * config["embedding_radius"][i]
                        out_y = point.lat.deg + np.cos(angle) * config["embedding_radius"][i]
                        gp_points.append(
                            self.observer.get_data_over_area(
                                x_interval=out_x,
                                y_interval=out_y,
                                t_interval=[t_interval[0], t_interval[-1]],
                                throw_exceptions=False,
                            )
                            .interp(
                                time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                                lon=out_x,
                                lat=out_y,
                                method="linear",
                                kwargs={
                                    "fill_value": "extrapolate",
                                },
                            )[variables]
                            .to_array()
                            .to_numpy()
                            .reshape((len(variables),len(t_interval),-1))
                        )
                # Observer: var (0), t (1), xy (2)
                gp_map = np.concatenate(gp_points, axis=-1)
            if (
                "mag" in config["features"]["observer_variables"]
                and "dir" in config["features"]["observer_variables"]
            ):
                gp_map[0] = np.arctan2(gp_map[0], gp_map[1])
                gp_map[1] = np.linalg.norm(gp_map[0:1], axis=0)

            maps.append(gp_map)
        # Step 4: C Currents
        if config["features"]["currents_hindcast"]:
            current_map = (
                obs.forecast_data_source.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                    kwargs={
                        "fill_value": "extrapolate",
                    },
                )["water_v", "water_u"]
                .to_array()
                .to_numpy()
            )
            # XArray: var (0), t (1), x (2), y (3)
            maps.append(current_map)
        # Step 5: FC Currents
        if config["features"]["currents_forecast"]:
            t_interval = [
                obs.platform_state.date_time + datetime.timedelta(hours=h)
                for h in config["features"]["currents_forecast"]
            ]
            current_map = (
                obs.forecast_data_source.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                    kwargs={
                        "fill_value": "extrapolate",
                    },
                )["water_u", "water_v"]
                .to_array()
                .to_numpy()
            )
            # XArray: var (0), t (1), x (2), y (3)
            maps.append(current_map)
        # Step 6: True Error
        if config["features"]["true_error"]:
            t_interval = [
                obs.platform_state.date_time + datetime.timedelta(hours=h)
                for h in config["features"]["true_error"]
            ]
            current_fc = (
                obs.forecast_data_source.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                    kwargs={
                        "fill_value": "extrapolate",
                    },
                )
                .to_array()
                .to_numpy()
            )
            current_hc = (
                obs.forecast_data_source.get_data_over_area(
                    x_interval=x_interval,
                    y_interval=y_interval,
                    t_interval=[t_interval[0], t_interval[-1]],
                )
                .interp(
                    time=[np.datetime64(t.replace(tzinfo=None)) for t in t_interval],
                    lon=np.linspace(x_interval[0], x_interval[1], config["xy_width_points"]),
                    lat=np.linspace(y_interval[0], y_interval[1], config["xy_width_points"]),
                    method="linear",
                    kwargs={
                        "fill_value": "extrapolate",
                    },
                )
                .to_array()
                .to_numpy()
            )
            # XArray: var (0), t (1), x (2), y (3)
            error_map = current_fc - current_hc
            maps.append(error_map)

        map = np.concatenate(maps, axis=0)
        map = np.clip(map, -10000, 10000)
        map = np.nan_to_num(map)
        map = map.astype("float32")

        if "xy_width_degree" in config and "xy_width_points" in config:
            if config["major"] == "time":
                map = np.moveaxis(map, [0, 1, 2, 3], [1, 0, 2, 3])
            map = map.reshape(-1, *map.shape[-2:])

        return map.flatten() if config["flatten"] else map.squeeze()

    def get_meta_from_state(self, obs: ArenaObservation, problem, config) -> np.ndarray:
        features = []

        if "lon" in config:
            features += [obs.platform_state.lon.deg]
        if "lat" in config:
            features += [obs.platform_state.lon.deg]
        if "time" in config:
            features += [obs.platform_state.date_time.timestamp()]
        if "target_distance" in config:
            features += [problem.distance(obs.platform_state).deg]
        if "target_direction" in config:
            features += [problem.bearing(obs.platform_state)]
        if "hj_fc_direction" in config:
            features += [
                self.forecast_planner._get_action_from_plan(state=obs.platform_state).direction
            ]
        if "hj_hc_direction" in config:
            features += [
                self.hindcast_planner._get_action_from_plan(state=obs.platform_state).direction
            ]
        if "episode_time_in_h" in config:
            features += [problem.passed_seconds(obs.platform_state) / 3600]
        if "ttr_center" in config:
            features += [
                self.forecast_planner.interpolate_value_function_in_hours(
                    point=obs.platform_state.to_spatio_temporal_point()
                ).squeeze()
            ]

        return np.array(features)
