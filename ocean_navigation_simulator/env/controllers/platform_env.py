import gym
from typing import Tuple, Optional, Union, Text
import numpy as np
import gym
from gym import spaces
from gym.utils.seeding import RandomNumberGenerator


class PlatformEnv(gym.Env):
    """
    Platform Learning Environment
    # TODO: define ActType and ObsType
    """

    # should be set in SOME gym subclasses -> TODO: see if we need to
    metadata = {"render_modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]
    _np_random: Optional[RandomNumberGenerator] = None

    def __init__(self, problem: Problem):
        seed = 189
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (ActType): an action provided by the agent
        Returns:
            observation (ObsType): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): auxiliary diagnostic information
        """
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
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
        pass

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


