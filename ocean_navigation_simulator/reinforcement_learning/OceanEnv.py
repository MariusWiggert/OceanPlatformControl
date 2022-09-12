# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import pickle
import time
import socket
from types import SimpleNamespace
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import gym
import psutil

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory
from ocean_navigation_simulator.reinforcement_learning.OceanRewardFunction import OceanRewardFunction
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import OceanFeatureConstructor
from ocean_navigation_simulator.reinforcement_learning.scripts import cluster_utils
from ocean_navigation_simulator.reinforcement_learning.bcolors import bcolors


class OceanEnv(gym.Env):
    spec = SimpleNamespace(id='OceanEnv', max_episode_steps=10000)
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        worker_index: Optional[int] = 0,
        config: Optional[dict] = {},
        verbose: Optional[int] = 0,
    ):
        with cluster_utils.timing(f'OceanEnv[Worker {worker_index}]: Created ({{:.1f}}s)', verbose):
            self.worker_index = worker_index
            self.config = config
            self.verbose = verbose
            self.private_ip = socket.gethostbyname(socket.gethostname())
            self.results_folder = f'{self.config["experiments_folder"]}/workers/worker {worker_index} ({self.private_ip})/'
            self.steps = None
            self.resets = 0

            self.random = np.random.default_rng(self.worker_index)
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

    def reset(self) -> np.array:
        self.reset_start_time = time.time()

        if self.steps is None or self.steps > 0:
            self.steps = 0
            self.rewards = []
            self.resets += 1
            self.problem = self.problem_factory.next_problem()
            self.prev_obs = self.arena.reset(self.problem.start_state)
            self.hindcast_planner = HJReach2DPlanner.from_saved_planner_state(
                folder=f'{self.config["generation_folder"]}groups/group_{self.problem.extra_info["group"]}/batch_{self.problem.extra_info["batch"]}/',
                problem=self.problem,
                verbose=self.verbose-1,
            )
            self.feature_constructor = OceanFeatureConstructor(
                planner=self.hindcast_planner,
                config=self.config['feature_constructor_config'],
                verbose=self.verbose-1
            )
            self.reward_function = OceanRewardFunction(
                planner=self.hindcast_planner,
                config=self.config['reward_function_config'],
                verbose=self.verbose-1
            )
            if self.config['fake'] == 'naive':
                self.naive_controller = NaiveController(problem=self.problem, verbose=self.verbose-1)
            if self.verbose > 0:
                print('OceanEnv[Worker {w}, Reset {r}]: Reset to Group {g} Batch {b} Index {i} ({t:.1f}s)'.format(
                    w=self.worker_index,
                    r=self.resets,
                    g=self.problem.extra_info["group"],
                    b=self.problem.extra_info["batch"],
                    i=self.problem.extra_info["factory_index"],
                    t=time.time()-self.reset_start_time,
                ))

        self.reset_end_time = time.time ()

        return self.feature_constructor.get_features_from_state(observation=self.prev_obs, problem=self.problem)

    def step(self, action: np.ndarray) -> Tuple[type(OceanFeatureConstructor.get_observation_space), float, bool, dict]:
        try:
            step_start = time.time()

            self.steps += 1

            if self.config['fake'] == 'random':
                platform_action = PlatformAction(magnitude=1, direction=self.random.integers(self.config['actions']) * 2 * np.pi / self.config['actions'])
            elif self.config['fake'] == 'naive':
                platform_action = self.naive_controller.get_action(observation=self.prev_obs)
            elif self.config['fake'] == 'hj_planner':
                platform_action = self.hindcast_planner.get_action(observation=self.prev_obs)
            elif isinstance(action, np.ndarray):
                platform_action = PlatformAction(magnitude=1, direction=action[0] * 2 * np.pi / self.config['actions'])
            elif isinstance(action, PlatformAction):
                platform_action = action
            else:
                platform_action = PlatformAction(magnitude=1, direction=action * 2 * np.pi / self.config['actions'])

            for i in range(self.config['arena_steps_per_env_step']):
                observation = self.arena.step(platform_action)

            problem_status = self.arena.problem_status(self.problem, check_inside=True, margin=self.config['feature_constructor_config']['ttr']['xy_width_degree']/2)
            done = problem_status != 0
            features = self.feature_constructor.get_features_from_state(observation=observation if not done else self.prev_obs, problem=self.problem)
            reward = self.reward_function.get_reward(self.prev_obs, observation if not done else self.prev_obs, self.problem, problem_status)
            self.rewards.append(reward)

            if done:
                if self.verbose > 0:
                    render_start = time.time()
                    if self.config['render']:
                        self.render()
                    print("OceanEnv[Worker {w}, Reset {r}]: Finished Group {g} Batch {b} Index {i} ({su}, {st} steps, {t:.1f}h, ∑ΔTTR {rew:.1f}h, %TTR {ttr:.1f}) (step Ø: {stt:.1f}ms, episode: {ep:.1f}s, reset: {res:.1f}s, render: {ren:.1f}s, Mem: {mem:,.0f}MB)".format(
                        w=self.worker_index,
                        r=self.resets,
                        g=self.problem.extra_info["group"],
                        b=self.problem.extra_info["batch"],
                        i=self.problem.extra_info["factory_index"],
                        su=f"{bcolors.OKGREEN}Success{bcolors.ENDC}" if problem_status > 0 else (f"{bcolors.FAIL}Timeout{bcolors.ENDC}" if problem_status == -1 else (f"{bcolors.FAIL}Stranded{bcolors.ENDC}" if problem_status == -2 else f"{bcolors.FAIL}Outside Arena{bcolors.ENDC}")),
                        st=self.steps,
                        t=self.problem.passed_seconds(observation.platform_state) / 3600,
                        rew=sum(self.rewards),
                        ttr=self.hindcast_planner.interpolate_value_function_in_hours(observation=self.prev_obs),
                        stt=1000 * (time.time() - self.reset_start_time) / self.steps,
                        ep=time.time() - self.reset_start_time,
                        res=self.reset_end_time - self.reset_start_time,
                        ren=time.time() - render_start,
                        mem=psutil.Process().memory_info().rss / 1e6
                    ))
            else:
                if self.verbose > 1:
                    print("OceanEnv[Worker {w}, Reset {r}]: Step {st} @ {ph:.0f}h {pm:.0f}min (step: {t:.1f}ms, {mem:,.0f}MB)".format(
                        w=self.worker_index,
                        r=self.resets,
                        st=self.steps,
                        ph=self.problem.passed_seconds(observation.platform_state)//3600,
                        pm=self.problem.passed_seconds(observation.platform_state)%3600/60,
                        t=1000*(time.time()-step_start),
                        mem=psutil.Process().memory_info().rss / 1e6,
                    ))

            self.prev_obs = observation
        except Exception as e:
            print(f'{bcolors.FAIL}{e}{bcolors.ENDC}')

        return features, reward, done, {
            'problem_status': problem_status,
            'arrival_time_in_h': self.problem.passed_seconds(observation.platform_state)/3600,
            'ram_usage_MB': int(psutil.Process().memory_info().rss / 1e6),
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
        cluster_utils.ensure_storage_connection()
        os.makedirs(self.results_folder, exist_ok=True)
        ax.get_figure().savefig(f'{self.results_folder}Reset {self.resets} Group {self.problem.extra_info["group"]} Batch {self.problem.extra_info["batch"]} Index {self.problem.extra_info["factory_index"]}.png')
        plt.clf()
