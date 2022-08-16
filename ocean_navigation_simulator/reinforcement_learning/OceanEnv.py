import time
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import gym

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.FeatureConstructor import FeatureConstructor
from ocean_navigation_simulator.environment.RewardFunction import RewardFunction
from ocean_navigation_simulator.environment.ProblemFactory import ProblemFactory

from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory
from ocean_navigation_simulator.reinforcement_learning.OceanRewardFunction import OceanRewardFunction
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import OceanFeatureConstructor


class OceanEnv(gym.Env):
    """
    A basic Platform Learning Environment
    """
    def __init__(
        self,
        config: Optional[dict] = {},
        verbose: Optional[int] = 0,
    ):
        """
        Constructs a basic Platform Learning Environment.

        actions: [magnitude, direction]
        observation: [abs lon, lon, time, u_curr, v_curr]

        Args:
            seed: PRNG seed for the environment
        """
        # if self.verbose > 0:
        print(f'OceanEnv: Created new Environment')
        self.config = {
            'seed': None,
            'experiments_path': '/seaweed-storage/experiments/test3',
            'scenario_name': 'gulf_of_mexico_HYCOM_hindcast',
            'generation_path': '/seaweed-storage/generation_runner/test3/',
            'arena_steps_per_env_step': 1,
            'actions': 8,
            'feature_constructor_config': {},
        } | config
        self.verbose = verbose

        self.action_space = gym.spaces.Discrete(self.config['actions'])
        self.observation_space = OceanFeatureConstructor.get_observation_space(self.config['feature_constructor_config'])
        self.reward_range = OceanRewardFunction.get_reward_range()

        self.problem_factory = FileMissionProblemFactory(seed=self.config['seed'], csv_file=f'{self.config["generation_path"]}problems.csv')

    def reset(self) -> np.array:
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        start = time.time()

        self.problem = self.problem_factory.next_problem()
        self.arena = ArenaFactory.create(scenario_name=self.config['scenario_name'], problem=self.problem, verbose=self.verbose-1)
        self.prev_obs = self.arena.reset(self.problem.start_state)
        self.hindcast_planner = HJReach2DPlanner(
            problem=self.problem,
            verbose=self.verbose-1
        )
        self.hindcast_planner.load_plan(
            f'/seaweed-storage/generation_runner/gulf_of_mexico_HYCOM_hindcast/seed_{self.problem.extra_info["seed"]}/batch_{self.problem.extra_info["batch"]}/'
        )
        self.feature_constructor = OceanFeatureConstructor(planner=self.hindcast_planner, verbose=self.verbose-1)
        self.reward_function = OceanRewardFunction(planner=self.hindcast_planner, verbose=self.verbose-1)
        features = self.feature_constructor.get_features_from_state(observation=self.prev_obs, problem=self.problem)

        if self.verbose > 0:
            print(f'OceanEnv: Reset ({time.time()-start:.1f}s)')

        return features

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if isinstance(action, np.ndarray):
            action = PlatformAction(magnitude=1, direction=action[0] * 2 * np.pi / self.config['actions'])
        elif isinstance(action, int):
            action = PlatformAction(magnitude=1, direction=action  * 2 * np.pi / self.config['actions'])

        for i in range(self.config['arena_steps_per_env_step']):
            observation = self.arena.step(action)

        problem_status = self.arena.problem_status(self.problem)

        features = self.feature_constructor.get_features_from_state(observation=observation, problem=self.problem)
        reward = self.reward_function.get_reward(self.prev_obs, observation, self.problem, problem_status)
        done = problem_status != 0

        self.prev_obs = observation

        return features, reward, done, { 'problem_status': problem_status, 'target_distance': self.problem.distance(observation.platform_state) }

    def render(self, mode="human"):
        fig, ax = plt.figure()
        self.arena.plot_all_on_map(ax=ax, problem=self.problem)
        fig.savefig()