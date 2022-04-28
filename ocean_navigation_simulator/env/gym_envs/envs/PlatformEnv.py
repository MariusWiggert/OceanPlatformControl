import time
from typing import Tuple, Optional, Union, Text, Callable, TypeVar
import numpy as np
import gym
from gym import spaces
from gym.utils.seeding import RandomNumberGenerator

from ocean_navigation_simulator.env.Platform import PlatformAction
from ocean_navigation_simulator.env.PlatformState import PlatformState
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.env.Arena import ArenaObservation, Arena

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class PlatformEnv(gym.Env):
    """
    A basic Platform Learning Environment
    """

    # should be set in SOME gym subclasses -> TODO: see if we need to
    metadata = {"render_modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]
    _np_random: Optional[RandomNumberGenerator] = None

    def __init__(self, problem_factory: ProblemFactory, arena: Arena = None,
                 reward_fn, feature_constructor, seed: Optional[int] = None):
        """
        Constructs a basic Platform Learning Environment.

        actions: [magnitude, direction]
        observation: [lat, lon, time, u_curr, v_curr]

        Args:
            problem_factory: yields next problem when reset() is called.
            arena: a platform arena to wrap, if None uses default arena
            seed: PRNG seed for the environment
            reward_fn: function that takes in some arguments and returns the rewards
            feature_constructor: function that takes in obs and other args and featurizes obs for model
        """
        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 6.3]), shape=(2,))
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(5,)) # TODO: consider changing bounds?

        self.problem = None
        self.prev_state = None

        self.problem_factory = problem_factory

        self.arena = arena

        self.reward_fn = reward_fn
        self.feature_constructor = feature_constructor

        self.reset()

    def step(self, action: PlatformAction) -> Tuple[np.ndarray, float, bool, dict]:
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
        command = PlatformAction(magnitude=action[0], direction=action[1]) # TODO: This constructor could be in another file
        arena_obs = self.arena.step(command)

        done = self.problem.is_done(arena_obs.platform_state)
        target = self.problem.end_region
        reward = self.reward_fn(self.prev_state, arena_obs.platform_state, target, done)

        if done:
            self.reset()

        model_obs = self.feature_constructor(arena_obs, target)
        self.prev_state = arena_obs.platform_state

        return model_obs, reward, done, None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ArenaObservation, tuple[ArenaObservation, dict]]:
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

        arena_obs = self.arena.reset()
        self.prev_state = arena_obs.platform_state
        model_obs = self.feature_constructor(arena_obs)

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


