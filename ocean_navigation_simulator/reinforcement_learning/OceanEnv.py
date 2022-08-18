import os
import pickle
import time
from types import SimpleNamespace
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import gym
import psutil

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformAction

from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory
from ocean_navigation_simulator.reinforcement_learning.OceanRewardFunction import OceanRewardFunction
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import OceanFeatureConstructor
from ocean_navigation_simulator.scripts.RayUtils import RayUtils
from ocean_navigation_simulator.utils.bcolors import bcolors


class OceanEnv(gym.Env):
    """
    A basic Platform Learning Environment
    """
    spec = SimpleNamespace(id='OceanEnv', max_episode_steps=10000)
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        worker_index: Optional[int] = 0,
        config: Optional[dict] = {},
        verbose: Optional[int] = 0,
    ):
        init_start = time.time()

        self.worker_index = worker_index
        self.config = config
        self.verbose = verbose
        self.results_folder = f'{self.config["experiments_folder"]}/workers/worker {worker_index}/'
        self.steps = None
        self.resets = 0

        self.action_space = gym.spaces.Discrete(self.config['actions'])
        self.observation_space = OceanFeatureConstructor.get_observation_space(self.config['feature_constructor_config'])
        self.reward_range = OceanRewardFunction.get_reward_range(self.config['reward_function_config'])
        self.problem_factory = FileMissionProblemFactory(seed=self.worker_index, csv_file=f'{self.config["generation_folder"]}problems.csv')
        with open(f'{self.config["generation_folder"]}config.pickle', 'rb') as file:
            self.problem_config = pickle.load(file)
        self.arena = ArenaFactory.create(
            scenario_name=self.config['scenario_name'],
            x_interval=self.problem_config['x_range'],
            y_interval=self.problem_config['y_range'],
            t_interval=self.problem_config['t_range'],
            verbose=self.verbose-1
        )

        if self.verbose > 0:
            print(f'OceanEnv[Worker {self.worker_index}]: Created ({time.time()-init_start:.1f}s)')

    def reset(self) -> np.array:
        if self.steps == 0:
            return self.feature_constructor.get_features_from_state(observation=self.prev_obs, problem=self.problem)

        start = time.time()

        self.resets += 1
        self.steps = 0
        self.reset_time = time.time()
        self.problem = self.problem_factory.next_problem()
        self.prev_obs = self.arena.reset(self.problem.start_state)
        self.hindcast_planner = HJReach2DPlanner.from_plan(
            folder=f'{self.config["generation_folder"]}seed_{self.problem.extra_info["seed"]}/batch_{self.problem.extra_info["batch"]}/',
            problem=self.problem,
            verbose=self.verbose-1,
        )
        self.feature_constructor = OceanFeatureConstructor(planner=self.hindcast_planner, config=self.config['feature_constructor_config'], verbose=self.verbose-1)
        self.reward_function = OceanRewardFunction(planner=self.hindcast_planner, verbose=self.verbose-1)
        features = self.feature_constructor.get_features_from_state(observation=self.prev_obs, problem=self.problem)

        if self.verbose > 0:
            print(f'OceanEnv[Worker {self.worker_index}, Reset {self.resets}]: Reset to Seed {self.problem.extra_info["seed"]} Batch {self.problem.extra_info["batch"]} ({time.time()-start:.1f}s)')

        return features

    def step(self, action: np.ndarray) -> Tuple[type(OceanFeatureConstructor.get_observation_space), float, bool, dict]:
        try:
            step_start = time.time()

            self.steps += 1

            if isinstance(action, np.ndarray):
                platform_action = PlatformAction(magnitude=1, direction=action[0] * 2 * np.pi / self.config['actions'])
            else:
                platform_action = PlatformAction(magnitude=1, direction=action * 2 * np.pi / self.config['actions'])

            for i in range(self.config['arena_steps_per_env_step']):
                observation = self.arena.step(platform_action)

            problem_status = self.arena.problem_status(self.problem, check_inside=False)
            features = self.feature_constructor.get_features_from_state(observation=observation, problem=self.problem)
            reward = self.reward_function.get_reward(self.prev_obs, observation, self.problem, problem_status)
            done = problem_status != 0

            self.prev_obs = observation

            if done:
                if self.verbose > 0:
                    render_start = time.time()
                    if self.config['render']:
                        self.render()
                    print("OceanEnv[Worker {}, Reset {}]: Finished Seed {} Batch {} ({}, {} steps, {:.1f}h) (Ø step time: {:.1f}ms, total time {:.1f}s, render: {}, Mem: {:,.0f}MB)".format(
                        self.worker_index,
                        self.resets,
                        self.problem.extra_info["seed"],
                        self.problem.extra_info["batch"],
                        f"{bcolors.OKGREEN}Success{bcolors.ENDC}" if problem_status > 0 else "Failure",
                        self.steps,
                        self.problem.passed_seconds(observation.platform_state) / 3600,
                        1000 * (time.time() - self.reset_time) / self.steps,
                        time.time() - self.reset_time,
                        time.time() - self.render_start,
                        psutil.Process().memory_info().rss / 1e6
                    ))
            else:
                if self.verbose > 1:
                    print(f'OceanEnv[{self.worker_index}]: Step {self.steps} @ {self.problem.passed_seconds(observation.platform_state)/3600:.1f}h ({1000*(time.time()-step_start):.1f}ms, {psutil.Process().memory_info().rss / 1e6:,.0f}MB)')
        except Exception as e:
            print(f'{bcolors.FAIL}{e}{bcolors.ENDC}')

        return features, reward, done, {
            'problem_status': problem_status,
            'target_distance': self.problem.distance(observation.platform_state),
            'env_time': time.time()-step_start,
            'step': self.steps,
        }

    def render(self, mode="human"):
        fig = plt.figure(figsize=(12,12))
        # ax = self.arena.plot_all_on_map(
        #     problem=self.problem,
        #     x_interval=[self.problem_config['x_range'][0].deg, self.problem_config['x_range'][1].deg],
        #     y_interval=[self.problem_config['y_range'][0].deg, self.problem_config['y_range'][1].deg]
        # )
        ax = self.arena.plot_all_on_map(
            problem=self.problem,
            x_interval=[self.hindcast_planner.grid.domain.lo[0], self.hindcast_planner.grid.domain.hi[0]],
            y_interval=[self.hindcast_planner.grid.domain.lo[1], self.hindcast_planner.grid.domain.hi[1]]
        )
        self.arena.plot_arena_frame_on_map(ax)
        self.hindcast_planner.plot_hj_frame(ax)
        ax.autoscale()
        # plt.title('')
        RayUtils.check_storage_connection()
        os.makedirs(self.results_folder, exist_ok=True)
        ax.get_figure().savefig(f'{self.results_folder}Reset {self.resets} Seed {self.problem.extra_info["seed"]} Batch {self.problem.extra_info["batch"]}.png')