import abc
import datetime
import numpy as np
from typing import Union, List
from scipy.stats import multivariate_normal
import abc
import datetime
import numpy as np
from typing import Union, List
from scipy.stats import multivariate_normal
import jax
import scipy
import jax.numpy as jnp
from copy import deepcopy

from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.Arena import PlatformAction
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import ParticleBelief
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleFilterObserver import ParticleFilterObserver
from ocean_navigation_simulator.controllers.pomdp_planners.GenerativeParticleFilter import GenerativeParticleFilter
from ocean_navigation_simulator.controllers.pomdp_planners.PFTDPWPlanner import PFTDPWPlanner
import time

class Dyn_Obs_Model_Position_Obs(abc.ABC):
    x_domain = [-0, 2]
    y_domain = [0, 1]

    def __init__(self, cov_matrix: np.array, u_max, dt, random_seed: int = None):
        self.random_seed = random_seed
        self.dt = dt
        self.u_max = u_max
        self.var = multivariate_normal(mean=[0, 0], cov=cov_matrix)

    def sample_new_action(self, action_set: set) -> int:
        """Sample a new action non-existent in the set
        Args:
            action_set: Set of actions already sampled
        Returns:
            action: Action as int
        """
        # Get the set of actions not sampled before
        missing_actions = set(np.arange(8)) - action_set

        # Sample random action from missing_actions
        return np.random.choice(tuple(missing_actions))

    # Note: if needed we can vectorize this function easily
    def sample_observation(self, states: np.array, actions: np.array) -> np.array:
        """Sample a measurement from the observation model.
        Args:
            states: state vector (n, 6) with columns (x, y, t, A, epsilon, omega)
            actions: action vector (n, 1) with columns (action)
        Returns:
            z: measurement vector (u_current, v_current) as np.array (n, 2)
        """
        # Get next states
        next_state = self.get_next_states(states, actions)
        # select only the X, Y positions
        position_next_state = next_state[:, :2]
        # add noise according to the covariance matrix
        return position_next_state + self.var.rvs(size=position_next_state.shape[0], random_state=self.random_seed)

    def evaluate_observations(self, states: np.array, observations: np.array) -> float:
        """Evaluate the probability of a measurement given a state.
        Args:
            states: state vector (n, 6) with columns (x, y, t, A, epsilon, omega)
            observations: measurement vector (n, 2) with columns (x_gps, y_gps) as np.array
        Returns:
            p: probability of the measurement given the state (n, 1) as np.array
        """
        # Get estimated currents in u and v direction for the states/particles
        error = observations - states[:, :2]
        # evaluate the probability of the observations
        return self.var.pdf(error)

    def get_next_states(self, states: np.array, actions: np.array) -> np.array:
        """Shallow state wrapper for dynamics model. This is deterministic!
        Input:
            s: (n, 6) numpay array of state with columns [x, y, t, A, epsilon, omega]
            actions: array (n,) of action integers (between 0 and 7 for directions. 0 action is 0*pi and every int after +pi/4)
        Output:
            s_next: numpy array of next states (n, 6) with columns [x, y, t, A, epsilon, omega]
        """
        # Get true currents in u and v direction
        curs = self.currents_analytical(states=states)
        dx = (self.u_max * np.cos(actions * np.pi / 4) + curs[:, 0]) * self.dt
        dy = (self.u_max * np.sin(actions * np.pi / 4) + curs[:, 1]) * self.dt

        new_states = states + np.array([dx, dy, self.dt * np.ones(dx.shape), np.zeros(dx.shape), np.zeros(dx.shape), np.zeros(dx.shape)]).T

        return new_states

    def currents_analytical(self, states: np.array) -> np.array:
        """Analytical Formula for u and v currents of Periodic Double Gyre.
        Note: this can be used for floats as input or np.arrays which are all the same shape.
        Args:
            states: numpy array of shape (n, 6) with the following columns: [lon, lat, posix_time, A, epsilon, omega]
        Returns:
            currents  data as numpy array (n, 2 with columns [u_current, v_current])
        """
        A, epsilon, omega = states[:, 3], states[:, 4], states[:, 5]

        a = epsilon * np.sin(omega * states[:, 2])
        b = 1 - 2 * a
        f = a * np.power(states[:, 0], 2) + b * states[:, 0]
        df_dx = 2 * a * states[:, 0] + b

        u_cur_out = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * states[:, 1])
        v_cur_out = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * states[:, 1]) * df_dx
        curr_out = np.array([u_cur_out, v_cur_out]).T

        return np.where(self.is_boundary(lon=states[:, 0], lat=states[:, 1]), np.zeros(shape=curr_out.shape), curr_out)

    def is_boundary(
        self,
        lon: Union[float, np.array],
        lat: Union[float, np.array],
    ) -> Union[float, np.array]:
        """Helper function to check if a state is in the boundary."""
        x_boundary = np.logical_or(
            lon < self.x_domain[0],
            lon > self.x_domain[1],
        )
        y_boundary = np.logical_or(
            lat < self.y_domain[0],
            lat > self.y_domain[1],
        )

        return np.logical_or(x_boundary, y_boundary).reshape([-1, 1])


class BaseDynObsModelParticles:
    # Note: the 0 action means no action!
    # Note: assumes particle_vel_func is vectorized already!
    def __init__(self, obs_noise: float, F_max: float, dt_dynamics: float, particle_vel_func,
                 key: int, n_actions: int, n_states: int, n_euler_per_dt_dynamics: int = 10):
        self._key = key
        self.dt_dynamics = dt_dynamics
        self.dt_euler = self.dt_dynamics / n_euler_per_dt_dynamics
        self.n_euler_per_dt_dynamics = n_euler_per_dt_dynamics
        self.F_max = F_max
        self.var = multivariate_normal(mean=[0, 0], cov=np.eye(2) * obs_noise)
        self.n_actions = n_actions
        self.n_states = n_states
        self.particle_vel_func = particle_vel_func

    def sample_new_action(self, action_set: set) -> int:
        """Sample a new action non-existent in the set
        Args:
            action_set: Set of actions already sampled
        Returns:
            action: Action as int
        """
        # Get the set of actions not sampled before
        missing_actions = set(np.arange(self.n_actions)) - action_set

        # Sample random action from missing_actions
        key, self._key = jax.random.split(self._key)
        np.random.seed(key)
        return np.random.choice(tuple(missing_actions))

    def sample_observation(self, states: np.array, actions: np.array) -> np.array:
        """Get the next position observation (including dynamics step forward)
        Args:
            states: state vector (n, n_states + n_params) with columns (x, y, t, A, epsilon, omega)
            actions: action vector (n, 1) with columns (action)
        Returns:
            z: position measurement vector (x_pos, y_pos) as np.array (n, 2)
        """
        # Get next states WITHOUT overwriting the old states using deepcopy
        next_state = deepcopy(states)
        next_state = self.get_next_states(next_state, actions)

        # select only the X, Y positions
        position_next_state = next_state[:, :2]

        # add noise to the position with variance self.obs_noise
        key, self._key = jax.random.split(self._key)
        return position_next_state # + self.var.rvs(size=position_next_state.shape[0], random_state=key[0].__int__())

    def evaluate_observations(self, states: np.array, observations: np.array) -> float:
        """Evaluate the probability of a measurement given a state.
        Args:
            states: state vector (n, 6) with columns (x, y, t, A, epsilon, omega)
            observations: measurement vector (n, 2) with columns (x_gps, y_gps) as np.array
        Returns:
            p: probability of the measurement given the state (n, 1) as np.array
        """
        # Get estimated currents in u and v direction for the states/particles
        error = observations - states[:, :self.n_states]
        # evaluate the probability of the observations
        return self.var.pdf(error)

    def get_next_states(self, states: np.array, actions: np.array) -> np.array:
        """Shallow state wrapper for dynamics model. This is deterministic and vectorized.
        Input:
            s: (n, n_states + n_params) numpay array of particles e.g. [x, y, t, A, epsilon, omega]
            actions: array (n,) of action integers (between 0 and 7 for directions. 0 action is 0*pi and every int after +pi/4)
        Output:
            s_next: numpy array of next states (n, 6) with columns [x, y, t, A, epsilon, omega]
        """

        # Note: this assumes the action is fixed over dt_dynamics
        angle = actions * np.pi / 4
        thrust = np.where(actions == 8, 0, self.F_max)
        actuation = thrust.reshape(-1, 1) * np.array([np.cos(angle), np.sin(angle)]).T

        # Take n_euler_per_dt_dynamics euler steps of size dt_dynamics/n_euler_per_dt_dynamics
        # TODO: this could be a compiled jax function.
        for _ in range(self.n_euler_per_dt_dynamics):
            states[:, :self.n_states] = states[:, :self.n_states] + self.dt_euler * (self.particle_vel_func(states) + actuation)

        # add time-step to time
        states[:, 2] = states[:, 2] + self.dt_dynamics

        return states


# Note: this uses jax a lot, not sure if it helps at all though...
class BaseDynObsModelParticlesJax:
    # Note: the 0 action means no action!

    def __init__(self, obs_noise: float, F_max: float, dt_dynamics: float, vel_func, key: int, n_actions: int, n_states: int):
        self._key = key
        self.dt_dynamics = dt_dynamics
        self.F_max = F_max
        self.obs_noise = obs_noise
        self.n_actions = n_actions
        self.n_states = n_states

        self.vel_mc_ode = lambda t, x, params: vel_func(t, *x, *params)
        self.vel_mc_ode = jax.vmap(self.vel_mc_ode, (None, 0, 0), 0)

    # @abstractmethod
    # @jax.jit
    def __call__(self, t, x, action, stoch_params):
        x = x.reshape([-1, 2])
        angle = (action - 1) * np.pi / 4
        thrust = np.where(action == 0, 0, self.F_max)
        act = jnp.stack([thrust * np.cos(angle), thrust * np.sin(angle)], axis=-1)
        return (self.vel_mc_ode(t, x, stoch_params) + act).ravel()

    # @jax.jit
    def sample_new_action(self, action_set: set) -> int:
        """Sample a new action non-existent in the set
        Args:
            action_set: Set of actions already sampled
        Returns:
            action: Action as int
        """
        # Get the set of actions not sampled before
        missing_actions = set(np.arange(self.n_actions)) - action_set

        # Sample random action from missing_actions
        key, self._key = jax.random.split(self._key)
        np.random.seed(key)
        return np.random.choice(tuple(missing_actions))

    # @jax.jit
    def sample_observation(self, states: jnp.array, actions: jnp.array) -> jnp.array:
        """Get the next position observation (including dynamics step forward)
        Args:
            states: state vector (n, n_states + n_params) with columns (x, y, t, A, epsilon, omega)
            actions: action vector (n, 1) with columns (action)
        Returns:
            z: position measurement vector (x_pos, y_pos) as np.array (n, 2)
        """
        # Get next states
        next_state = self.get_next_states(states,actions)

        # select only the X, Y positions
        position_next_state = next_state[:, :2]

        # add noise to the position with variance self.obs_noise
        key, self._key = jax.random.split(self._key)
        return position_next_state + self.obs_noise * jax.random.normal(key, shape=position_next_state.shape)

    # @jax.jit
    def evaluate_observations(self, states: jnp.array, observations: jnp.array) -> float:
        """Evaluate the probability of a measurement given a state.
        Args:
            states: state vector (n, 6) with columns (x, y, t, A, epsilon, omega)
            observations: measurement vector (n, 2) with columns (x_gps, y_gps) as np.array
        Returns:
            p: probability of the measurement given the state (n, 1) as np.array
        """
        # Get the error between the position observation(s) and the estimated state(s)
        error = observations - states[:, :2]

        # evaluate the probability of the observations
        return jax.scipy.stats.multivariate_normal.pdf(error, mean=jnp.array([0,0]), cov=jnp.eye(2) * self.obs_noise)

    def get_next_states(self, states: jnp.array, actions: jnp.array) -> np.array:
        """Shallow state wrapper for dynamics model. This is deterministic and vectorized.
        Input:
            s: (n, n_states + n_params) numpay array of particles e.g. [x, y, t, A, epsilon, omega]
            actions: array (n,) of action integers (between 0 and 7 for directions. 0 action is 0*pi and every int after +pi/4)
        Output:
            s_next: numpy array of next states (n, 6) with columns [x, y, t, A, epsilon, omega]
        """

        # Note: this assumes the action is fixed over dt_dynamics
        def rhs(t, x):
            return self(t, x, action=actions, stoch_params=states[:, self.n_states:])

        int_solve = scipy.integrate.solve_ivp(
            fun=rhs,  # this is the right hand side of the ODE
            t_span=[states[0, 2], states[0, 2] + self.dt_dynamics],   # solve from t to t+dt
            y0=states[:, :2].ravel(),   # start at x, y
            dense_output=False, events=None, vectorized=False)

        trajs = int_solve.y.reshape([*states[:, :2].shape, -1])
        next_position_states = trajs[..., -1]

        # # simple euler step
        # euler_step_solve = states[:, :2].ravel() + self.dt_dynamics * rhs(t=states[0, 2], x=states[:, :2].ravel())
        # next_position_states = euler_step_solve.reshape([*states[:, :2].shape])

        return jnp.concatenate((next_position_states, states[:, self.n_states - 1].reshape(-1, 1) + self.dt_dynamics, states[:, self.n_states:]), axis=1)