from typing import Optional, Text, Tuple, Union

import gym
import numpy as np

from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.FeatureConstructor import (
    FeatureConstructor,
)
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.PlatformState import PlatformState
from ocean_navigation_simulator.environment.ProblemFactory import (
    ProblemFactory,
)
from ocean_navigation_simulator.problem_factories.DoubleGyreProblemFactory import (
    DoubleGyreProblemFactory,
)
from ocean_navigation_simulator.reinforcement_learning.DoubleGyreFeatureConstructor import (
    DoubleGyreFeatureConstructor,
)
from ocean_navigation_simulator.reinforcement_learning.DoubleGyreRewardFunction import (
    DoubleGyreRewardFunction,
)
from ocean_navigation_simulator.reinforcement_learning.RewardFunction import (
    RewardFunction,
)


class DoubleGyreEnv(gym.Env):
    """
    A basic Platform Learning Environment
    """

    metadata = {"render_modes": []}
    reward_range: Tuple = None
    observation_space: gym.spaces.Box = None
    spec = None  # might use for max_episode_steps

    problem_factory: ProblemFactory = None
    problem: NavigationProblem = None
    arena: Arena = None
    prev_state: PlatformState = None
    reward_function: RewardFunction = None
    feature_constructor: FeatureConstructor = None

    # _np_random: Optional[RandomNumberGenerator] = None

    def __init__(
        self,
        config,
        scenario_name="double_gyre",
    ):
        """
        Constructs a basic Platform Learning Environment.

        actions: [magnitude, direction]
        observation: [abs lon, lon, time, u_curr, v_curr]

        Args:
            seed: PRNG seed for the environment
        """
        self.seed = config["seed"] if "seed" in config else 2022
        self.arena_steps_per_env_step = (
            config["arena_steps_per_env_step"] if "arena_steps_per_env_step" in config else 1
        )

        self.arena = ArenaFactory.create(scenario_name=scenario_name)
        self.problem_factory = DoubleGyreProblemFactory(seed=self.seed, scenario_name="simplified")
        self.reward_function = DoubleGyreRewardFunction()
        self.feature_constructor = DoubleGyreFeatureConstructor()

        self.action_space = gym.spaces.Box(
            low=np.array([-np.pi], dtype=np.float32),
            high=np.array([np.pi], dtype=np.float32),
            shape=(1,),
        )
        # self.action_space = gym.spaces.Discrete(360)
        self.reward_range = self.reward_function.get_rewarrd_range()
        self.observation_space = self.feature_constructor.get_observation_space()

        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (SimulatorAction): an action provided by the agent
        Returns:
            observation (SimulatorObservation): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): auxiliary diagnostic information
        """
        command = PlatformAction(magnitude=1, direction=action[0])

        for i in range(self.arena_steps_per_env_step):
            arena_obs = self.arena.step(command)

        problem_status = self.problem.is_done(arena_obs.platform_state)

        solved = problem_status == 1
        unsolvable = problem_status == -1
        crashed = not self.arena.is_inside_arena() or self.arena.is_on_land()
        terminate = solved or unsolvable or crashed

        reward = self.reward_function.get_reward(
            self.prev_state, arena_obs.platform_state, self.problem, solved, crashed
        )

        # if terminate:
        #     print('terminate')
        #     self.reset()

        model_obs = self.feature_constructor.get_features_from_state(
            obs=arena_obs, problem=self.problem
        )
        self.prev_state = arena_obs.platform_state

        return model_obs, reward, terminate, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """
        Resets the environment.
        This method should also reset the environment's random number
        generator(s) if `seed` is an integer or if the environment has not
        yet initialized a random number generator. If the environment already
        has a random number generator and `reset` is called with `seed=None`,
        the RNG should not be reset.
        Moreover, `reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.
        Args:
            options:
            seed: Seed for reseeding random number generators
            return_info: if True, return info dictionary
        Returns:
            observation (object): the initial observation.
            info (opt. dictionary): a dictionary containing extra information, returned if return_info set to True
        """
        self.problem = self.problem_factory.next_problem()

        arena_obs = self.arena.reset(self.problem.start_state)
        self.prev_state = arena_obs.platform_state
        model_obs = self.feature_constructor.get_features_from_state(
            obs=arena_obs, problem=self.problem
        )

        if return_info:
            pass
            # simulator_state = self.arena.get_simulator_state()
            # info = simulator_state.balloon_state
            # return observation, info
        else:
            return model_obs

    def render(self, mode="human") -> Union[None, np.ndarray, Text]:
        """
        Renders the environment.
        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render_modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Returns:
            None, a numpy array of rgb data, or a string (?) object, depending on the mode.
        """
        return self.problem.renderer.render(mode)
