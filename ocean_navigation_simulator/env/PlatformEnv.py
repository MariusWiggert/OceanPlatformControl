from typing import Tuple, Optional, Union, Text, Callable
import numpy as np
import gym

from ocean_navigation_simulator.env.Arena import Arena
from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.DoubleGyreProblemFactory import DoubleGyreProblemFactory
from ocean_navigation_simulator.env.FeatureConstructors import double_gyre_feature_constructor, double_gyre_simple_feature_constructor
from ocean_navigation_simulator.env.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.env.PlatformState import PlatformState
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.env.RewardFunctions import double_gyre_reward_function
from ocean_navigation_simulator.env.Platform import PlatformAction


class PlatformEnv(gym.Env):
    """
    A basic Platform Learning Environment
    """

    # should be set in SOME gym subclasses -> TODO: see if we need to
    metadata = {"render_modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    problem_factory: ProblemFactory = None
    problem: NavigationProblem = None
    arena: Arena = None
    prev_state: PlatformState = None
    reward_fn: Callable = None
    feature_constructor: Callable= None

    # _np_random: Optional[RandomNumberGenerator] = None

    def __init__(
        self,
        seed: Optional[int] = 2022,
        env_steps_per_arena_steps: Optional[int] = 10,
    ):
        """
        Constructs a basic Platform Learning Environment.

        actions: [magnitude, direction]
        observation: [abs lon, lon, time, u_curr, v_curr]

        Args:
            seed: PRNG seed for the environment
        """
        self.seed = seed
        self.env_steps_per_arena_steps = env_steps_per_arena_steps

        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([2 * np.pi]),
            shape=(1,)
        )  # TODO: consider normalization to improve training

        self.observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(3,)
        )  # TODO: consider changing bounds?

        self.problem_factory = DoubleGyreProblemFactory(seed=seed, scenario_name='simplified')
        self.arena = ArenaFactory.create(scenario_name='double_gyre')
        self.reward_fn = double_gyre_reward_function
        self.feature_constructor = double_gyre_simple_feature_constructor

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

        for i in range(self.env_steps_per_arena_steps):
            arena_obs = self.arena.step(command)

        problem_status = self.problem.is_done(arena_obs.platform_state)

        solved = problem_status == 1
        unsolvable = problem_status == -1
        crashed = not self.arena.is_inside_arena()
        terminate = solved or unsolvable or crashed

        reward = self.reward_fn(self.prev_state, arena_obs.platform_state, self.problem, solved, crashed)

        if terminate:
            self.reset()

        model_obs = self.feature_constructor(arena_obs, self.problem)
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
        model_obs = self.feature_constructor(arena_obs, self.problem)

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
