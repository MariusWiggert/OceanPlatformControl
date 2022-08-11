import os
from typing import Tuple, Optional, Union, Text
import numpy as np
import gym

from ocean_navigation_simulator.environment.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.FeatureConstructor import FeatureConstructor
from ocean_navigation_simulator.environment.RewardFunction import RewardFunction
from ocean_navigation_simulator.environment.ProblemFactory import ProblemFactory

from ocean_navigation_simulator.problem_factories.MissionProblemFactory import MissionProblemFactory
from ocean_navigation_simulator.reinforcement_learning.DoubleGyreFeatureConstructor import DoubleGyreFeatureConstructor
from ocean_navigation_simulator.reinforcement_learning.DoubleGyreRewardFunction import DoubleGyreRewardFunction
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import OceanFeatureConstructor


class OceanEnv(gym.Env):
    """
    A basic Platform Learning Environment
    """
    reward_range: Tuple = None
    action_space: gym.Space = None
    observation_space: gym.Space = None

    arena: Arena = None
    prev_obs: ArenaObservation = None
    problem: NavigationProblem = None

    problem_factory: ProblemFactory = None
    reward_function: RewardFunction = None
    feature_constructor: FeatureConstructor = None

    def __init__(
        self,
        config = {},
    ):
        """
        Constructs a basic Platform Learning Environment.

        actions: [magnitude, direction]
        observation: [abs lon, lon, time, u_curr, v_curr]

        Args:
            seed: PRNG seed for the environment
        """
        self.config = { 'seed': None, 'arena_steps_per_env_step': 1 } | config

        self.problem_factory = MissionProblemFactory(seed=self.config['seed'])
        self.feature_constructor = OceanFeatureConstructor()
        self.reward_function = DoubleGyreRewardFunction()

        self.action_space = gym.spaces.Discrete(8)
        self.reward_range = self.reward_function.get_reward_range()
        self.observation_space = self.feature_constructor.get_observation_space()

        self.reset()

    def reset(self):
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
        self.problem = self.problem_factory.next_problem()

        # self.arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast')
        # ArenaFactory.download_hycom_forecast(problem=self.problem, n_days_ahead=6)
        self.arena = ArenaFactory.create(scenario_name='gulf_of_mexico_Copernicus_forecast_and_hindcast')

        self.prev_obs = self.arena.reset(self.problem.start_state)

        return self.feature_constructor.get_features_from_state(obs=self.prev_obs, problem=self.problem)

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
        command = PlatformAction(magnitude=1, direction=action[0] / 4 * np.pi)

        for i in range(self.config['arena_steps_per_env_step']):
            arena_obs = self.arena.step(command)

        problem_status = self.arena.problem_status(self.problem)

        model_obs = self.feature_constructor.get_features_from_state(obs=arena_obs, problem=self.problem)
        reward = self.reward_function.get_reward(self.prev_obs.platform_state, arena_obs.platform_state, self.problem, problem_status)
        terminate = problem_status != 0

        self.prev_state = arena_obs.platform_state

        return model_obs, reward, terminate, {}


    def render(self, mode="human") -> Union[None, np.ndarray, Text]:
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        return self.problem.renderer.render(mode)
