import time
from typing import Tuple, Optional
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

from ocean_navigation_simulator.problem_factories.MissionProblemFactory import MissionProblemFactory
from ocean_navigation_simulator.reinforcement_learning.OceanRewardFunction import OceanRewardFunction
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
    hindcast_planner: HJReach2DPlanner = None

    problem_factory: ProblemFactory = None
    feature_constructor: FeatureConstructor = None
    reward_function: RewardFunction = None

    def __init__(
        self,
        env_config: Optional[dict] = {},
        scenario_name: Optional[str] = 'gulf_of_mexico_HYCOM_hindcast',
        config: Optional[dict] = {},
        verbose: Optional[bool] = False,
    ):
        """
        Constructs a basic Platform Learning Environment.

        actions: [magnitude, direction]
        observation: [abs lon, lon, time, u_curr, v_curr]

        Args:
            seed: PRNG seed for the environment
        """
        self.scenario_name = scenario_name
        self.config = { 'seed': None, 'arena_steps_per_env_step': 1 } | config
        self.verbose = verbose

        self.problem_factory = MissionProblemFactory(seed=self.config['seed'])
        self.action_space = gym.spaces.Discrete(8)

        self.observation_space = OceanFeatureConstructor.get_observation_space()
        self.reward_range = OceanRewardFunction.get_reward_range()

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
        if self.verbose:
            print(f'OceanEnv: Reset')

        self.problem = self.problem_factory.next_problem()

        self.arena = ArenaFactory.create(scenario_name=self.scenario_name, problem=self.problem, verbose=self.verbose)
        self.prev_obs = self.arena.reset(self.problem.start_state)

        start = time.time()
        specific_settings = {
            'replan_on_new_fmrc': True,
            'replan_every_X_seconds': None,
            'direction': 'multi-time-reach-back',
            'n_time_vector': 199,  # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
            'deg_around_xt_xT_box': 1.0,  # area over which to run HJ_reachability
            'accuracy': 'high',
            'artificial_dissipation_scheme': 'local_local',
            'T_goal_in_seconds': self.problem.timeout.total_seconds(),
            'use_geographic_coordinate_system': True,
            'progress_bar': self.verbose,
            'initial_set_radii': [0.1, 0.1],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
            # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
            'grid_res': 0.04,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
            'd_max': 0.0,
            # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
            # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
            'platform_dict': self.arena.platform.platform_dict
        }
        self.hindcast_planner = HJReach2DPlanner(problem=self.problem, specific_settings=specific_settings, verbose=self.verbose)
        if self.verbose:
            print(f'OceanEnv: Create HJReach2DPlanner ({time.time()-start:.1f}s)')

        start = time.time()
        self.feature_constructor = OceanFeatureConstructor(planner=self.hindcast_planner, verbose=self.verbose)
        if self.verbose:
            print(f'OceanEnv: Create OceanFeatureConstructor ({time.time()-start:.1f}s)')
        start = time.time()
        self.reward_function = OceanRewardFunction(planner=self.hindcast_planner, verbose=self.verbose)
        if self.verbose:
            print(f'OceanEnv: Create OceanRewardFunction ({time.time()-start:.1f}s)')

        return self.feature_constructor.get_features_from_state(observation=self.prev_obs, problem=self.problem)

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
            action = PlatformAction(magnitude=1, direction=action[0] / 4 * np.pi)
        elif isinstance(action, int):
            action = PlatformAction(magnitude=1, direction=action / 4 * np.pi)

        for i in range(self.config['arena_steps_per_env_step']):
            arena_obs = self.arena.step(action)

        problem_status = self.arena.problem_status(self.problem)

        features = self.feature_constructor.get_features_from_state(observation=arena_obs, problem=self.problem)
        reward = self.reward_function.get_reward(self.prev_obs, arena_obs, self.problem, problem_status)
        done = problem_status != 0

        self.prev_obs = arena_obs

        return features, reward, done, { 'problem_status': problem_status }