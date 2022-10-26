import logging
import os
import random
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from ocean_navigation_simulator.controllers.NaiveController import (
    NaiveController,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)
from ocean_navigation_simulator.reinforcement_learning.env.OceanFeatureConstructor import (
    OceanFeatureConstructor,
)
from ocean_navigation_simulator.reinforcement_learning.env.OceanRewardFunction import (
    OceanRewardFunction,
)
from ocean_navigation_simulator.utils import cluster_utils
from ocean_navigation_simulator.utils.misc import (
    bcolors,
    set_arena_loggers,
    timing,
    silence_ray_and_tf,
)

class OceanEnv(gym.Env):
    """
    OceanEnv encapsulates the simulation with a gym.Env interface for use in rllib
    - OceanEnv.init: initializes the arena, FeatureConstructor and RewardGenerator
    - OceanEnv.reset: runs arena.reset and loads hindcast & forecast planner
    - OceanEnv.step: runs arena.step, FeatureConstructor, and RewardGenerator
    """

    spec = SimpleNamespace(id="OceanEnv", max_episode_steps=10000)
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: dict,
        feature_constructor_config: dict,
        reward_function_config: dict,
        indices: Optional[List] = None,
        worker_index: Optional[int] = 0,
        result_folder: Optional[str] = None,
        verbose: Optional[int] = 0,
    ):
        with timing(
            f'OceanEnv[Worker {worker_index}]: Created with {"no" if indices is None else len(indices)} indices ({{:.1f}}s)',
            verbose,
        ):
            self.config = config
            self.feature_constructor_config = feature_constructor_config
            self.reward_function_config = reward_function_config
            self.indices = indices
            self.worker_index = worker_index
            self.result_folder = result_folder
            self.verbose = verbose

            # Step 1: Initialize Variables
            self.resets = 0

            if verbose > 1:
                set_arena_loggers(logging.INFO)
            else:
                set_arena_loggers(logging.WARNING)
                silence_ray_and_tf()

            # Step 2: Fix Seeds
            self.random = np.random.default_rng(self.worker_index)
            np.random.seed(self.worker_index)
            random.seed(self.worker_index)
            torch.manual_seed(self.worker_index)

            # Step 3: Set Env Description
            self.action_space = gym.spaces.Discrete(self.config["actions"])
            self.observation_space = OceanFeatureConstructor.get_observation_space(
                self.feature_constructor_config
            )
            self.reward_range = OceanRewardFunction.get_reward_range(self.reward_function_config)

            # Step 4: Initialize Problem Factory
            cluster_utils.ensure_storage_connection()
            self.problem_factory = FileProblemFactory(
                seed=self.worker_index,
                csv_file=f"{self.config['missions']['folder']}problems.csv",
                filter={"indices": indices},
            )
            # with open(f"{self.config['problem_folder']}config/config.pickle", "rb") as file:
            #     self.problem_config = pickle.load(file)

    def reset(self, *args) -> np.array:
        self.reset_start_time = time.time()

        # Step 1: Initialize Variables
        self.steps = 0
        self.resets += 1
        self.rewards = []

        # Step 2: Get Problem & Initialise Arena
        self.problem = self.problem_factory.next_problem()
        self.arena = ArenaFactory.create(
            scenario_file=self.config["scenario_file"],
            scenario_config=self.config["scenario_config"],
            x_interval=self.problem.extra_info["x_cache"],
            y_interval=self.problem.extra_info["y_cache"],
            problem=self.problem,
            verbose=self.verbose - 1,
        )
        self.prev_obs = self.arena.reset(self.problem.start_state)

        # Step 3: Load Cached Planners
        self.hindcast_planner = self.problem.get_cached_hindcast_planner(
            self.config["missions"]["folder"]
        )
        self.forecast_planner = self.problem.get_cached_forecast_planner(
            self.config["missions"]["folder"]
        )

        # Step 4: Initialize Feature Constructor, Reward Function & Controllers
        self.feature_constructor = OceanFeatureConstructor(
            forecast_planner=self.forecast_planner,
            hindcast_planner=self.hindcast_planner,
            config=self.feature_constructor_config,
        )
        self.reward_function = OceanRewardFunction(
            forecast_planner=self.forecast_planner,
            hindcast_planner=self.hindcast_planner,
            config=self.reward_function_config,
        )
        if self.config["fake"] == "naive":
            self.naive_controller = NaiveController(problem=self.problem)

        # Step 5:
        if self.verbose > 1:
            print(
                "OceanEnv[Worker {w}, Reset {r}]: Reset to (I{i}, G{gr} B{b} FI{fi}) ({t:.1f}s)".format(
                    w=self.worker_index,
                    r=self.resets,
                    i=self.problem.extra_info["index"],
                    gr=self.problem.extra_info["group"],
                    b=self.problem.extra_info["batch"],
                    fi=self.problem.extra_info["factory_index"],
                    t=time.time() - self.reset_start_time,
                )
            )

        self.reset_end_time = time.time()

        return self.feature_constructor.get_features_from_state(
            obs=self.prev_obs,
            problem=self.problem,
        )

    def step(
        self, action: int
    ) -> Tuple[type(OceanFeatureConstructor.get_observation_space), float, bool, dict]:
        step_start = time.time()
        self.steps += 1

        try:
            # Step 1: Get Action & Run Arena
            for i in range(self.config["arena_steps_per_env_step"]):
                if self.config["fake"] == "random":
                    platform_action = PlatformAction(
                        magnitude=1,
                        direction=self.random.integers(self.config["actions"])
                        * 2
                        * np.pi
                        / self.config["actions"],
                    )
                elif self.config["fake"] == "naive":
                    platform_action = self.naive_controller.get_action(observation=self.prev_obs)
                elif self.config["fake"] == "hj_planner_forecast":
                    platform_action = self.forecast_planner.get_action(
                        observation=self.prev_obs
                    )
                elif self.config["fake"] == "hj_planner_hindcast":
                    # Hindcast Planner has to receive hindcast_data_source
                    platform_action = self.hindcast_planner.get_action(
                        observation=self.prev_obs.replace_datasource(
                            self.arena.ocean_field.hindcast_data_source
                        )
                    )
                elif self.config["fake"] == "residual":
                    platform_action = self.forecast_planner.get_action(
                        observation=self.prev_obs
                    )
                    platform_action = PlatformAction(
                        magnitude=platform_action.magnitude,
                        direction=platform_action.direction
                        + action * 2 * np.pi / self.config["actions"],
                    )
                elif isinstance(action, PlatformAction):
                    platform_action = action
                else:
                    platform_action = PlatformAction(
                        magnitude=1, direction=action * 2 * np.pi / self.config["actions"]
                    )

                observation = self.arena.step(platform_action)

            # Step 2: Get Features & Rewards
            problem_status = self.arena.problem_status(self.problem)
            problem_status_text = self.arena.problem_status_text(problem_status)
            done = problem_status != 0

            if not done:
                self.forecast_planner.replan_if_necessary(observation)
                self.hindcast_planner.replan_if_necessary(
                    observation.replace_datasource(self.arena.ocean_field.hindcast_data_source)
                )

            features = self.feature_constructor.get_features_from_state(
                obs=(observation if not done else self.prev_obs),
                problem=self.problem,
            )
            reward = self.reward_function.get_reward(
                prev_obs=self.prev_obs,
                curr_obs=(observation if not done else self.prev_obs),
                problem=self.problem,
                problem_status=problem_status,
            )
            self.rewards.append(reward)
        except Exception:
            print(
                "OceanEnv[Worker {w}, Reset {r}]:".format(w=self.worker_index, r=self.resets),
                "Error at (I{i}, G{gr} B{b} FI{fi})".format(
                    i=self.problem.extra_info["index"],
                    gr=self.problem.extra_info["group"],
                    b=self.problem.extra_info["batch"],
                    fi=self.problem.extra_info["factory_index"],
                ),
            )
            raise

        # Step
        if done:
            if self.verbose > 0:
                render_start = time.time()
                if self.config["render"]:
                    self.render()
                print(
                    "OceanEnv[Worker {w}, Reset {r}]:".format(
                        w=self.worker_index,
                        r=self.resets,
                    ),
                    "Finished (I{i}, G{gr} B{b} FI{fi})".format(
                        i=self.problem.extra_info["index"],
                        gr=self.problem.extra_info["group"],
                        b=self.problem.extra_info["batch"],
                        fi=self.problem.extra_info["factory_index"],
                    ),
                    "({su}, {st} steps, {t:.1f}h, ttr@0: {ttr_0:.1f}h, ttr@t: {ttr_t:.1f}h, ∑rewards {rew:.1f}h, )".format(
                        su=bcolors.green(problem_status_text)
                        if problem_status > 0
                        else bcolors.red(problem_status_text),
                        st=self.steps,
                        t=self.problem.passed_seconds(observation.platform_state) / 3600,
                        ttr_0=self.problem.extra_info["ttr_in_h"],
                        ttr_t=self.hindcast_planner.interpolate_value_function_in_hours(
                            point=self.prev_obs.platform_state.to_spatio_temporal_point()
                        ),
                        rew=sum(self.rewards),
                    ),
                    "(step Ø: {stt:.1f}ms, total: {ep:.1f}s, reset: {res:.1f}s, render: {ren:.1f}s, Mem: {mem:,.0f}MB)".format(
                        stt=1000 * (time.time() - self.reset_start_time) / self.steps,
                        ep=time.time() - self.reset_start_time,
                        res=self.reset_end_time - self.reset_start_time,
                        ren=time.time() - render_start,
                        mem=psutil.Process().memory_info().rss / 1e6,
                    ),
                )
        else:
            if self.verbose > 2:
                print(
                    "OceanEnv[Worker {w}, Reset {r}]:".format(
                        w=self.worker_index,
                        r=self.resets,
                    ),
                    "Step {st} @ {ph:.0f}h, {pm:.0f}min, sim:{sim_time}".format(
                        st=self.steps,
                        ph=self.problem.passed_seconds(observation.platform_state) // 3600,
                        pm=self.problem.passed_seconds(observation.platform_state) % 3600 / 60,
                        sim_time=observation.platform_state.date_time,
                    ),
                    "(step: {t:.1f}ms, {mem:,.0f}MB)".format(
                        t=1000 * (time.time() - step_start),
                        mem=psutil.Process().memory_info().rss / 1e6,
                    ),
                )

        self.prev_obs = observation

        # Step X: Return Env Variables
        return (
            features,
            reward,
            done,
            {
                "problem_status": problem_status,
                "arrival_time_in_h": self.problem.passed_seconds(observation.platform_state) / 3600,
                "ram_usage_MB": int(psutil.Process().memory_info().rss / 1e6),
                "episode_time": time.time() - self.reset_start_time,
                "average_step_time": (time.time() - self.reset_start_time) / self.steps,
                "problem": self.problem,
                "episode_length": self.steps,
            },
        )

    def render(self, mode="human"):
        if self.result_folder is not None:
            # Step 1: Plot Arena & Frames
            ax = self.arena.plot_all_on_map(
                problem=self.problem,
                x_interval=[
                    self.hindcast_planner.grid.domain.lo[0],
                    self.hindcast_planner.grid.domain.hi[0],
                ],
                y_interval=[
                    self.hindcast_planner.grid.domain.lo[1],
                    self.hindcast_planner.grid.domain.hi[1],
                ],
                return_ax=True,
            )
            self.arena.plot_arena_frame_on_map(ax)
            self.hindcast_planner.plot_hj_frame(ax)
            ax.autoscale()

            # Step 2: Save Figure
            cluster_utils.ensure_storage_connection()
            os.makedirs(self.result_folder, exist_ok=True)
            ax.get_figure().savefig(
                f'{self.result_folder}Reset {self.resets} Group {self.problem.extra_info["group"]} Batch {self.problem.extra_info["batch"]} Index {self.problem.extra_info["factory_index"]}.png'
            )
            plt.clf()
        else:
            raise ValueError("result_folder not defined in OceanEnv")
