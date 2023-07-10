#%% 
# Dynamics and Observation Model for Double Gyre Flow Planning
# Note: it is vectorized to work with as many particle in parallel as needed.
import abc
import datetime
import numpy as np
from typing import Union, List
from scipy.stats import multivariate_normal

from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.Platform import PlatformAction
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import ParticleBelief
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleFilterObserver import ParticleFilterObserver

# update version including epsilon_sep in the state!
class DynamicsAndObservationModel(abc.ABC):
    x_domain = [-0, 2]
    y_domain = [0, 1]

    def __init__(self, cov_matrix: np.array, u_max: float = 0.2, dt: float = 0.5, random_seed: int = None):
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
    def sample_observation(self, states: np.array) -> np.array:
        """Sample a measurement from the observation model.
        Args:
            states: state vector (n, 6) with columns (x, y, t, A, epsilon, omega)
        Returns:
            z: measurement vector (u_current, v_current) as np.array (n, 2)
        """
        # Get true currents in u and v direction
        true_currents = self.currents_analytical(states=states)
        # add noise according to the covariance matrix
        return true_currents + self.var.rvs(size= true_currents.shape[0], random_state=self.random_seed)

    def evaluate_observations(self, states: np.array, observations: np.array) -> float:
        """Evaluate the probability of a measurement given a state.
        Args:
            states: state vector (n, 6) with columns (x, y, t, A, epsilon, omega)
            observations: measurement vector (n, 2) with columns (u_current, v_current) as np.array
        Returns:
            p: probability of the measurement given the state (n, 1) as np.array
        """
        # Get estimated currents in u and v direction for the states/particles
        true_currents = self.currents_analytical(states=states)
        error = observations - true_currents
        # evaluate the probability of the observations
        return self.var.pdf(error)

    def get_next_states(self, states: np.array, actions: np.array) -> np.array:
        """Shallow state wrapper for dynamics model.
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
#%%
# True parameters
A = 0.4
eps = 0.3
omega = 2*np.pi/1.5 # -> this means 1.5 time-units period time
F_max = 1.0
# Start - Goal Settings
# init_state = [0.3, 0.2, 0]
init_state = [1.0, 0.5, 0.3]
target_position = [1.7,0.8]
target_radius = 0.10

dt_sim = 0.1
timeout_of_simulation = 100     # in seconds

arenaConfig = {
    "casadi_cache_dict": {"deg_around_x_t": 0.5, "time_around_x_t": 10.0},
    "ocean_dict": {
        "hindcast": {
            "field": "OceanCurrents",
            "source": "analytical",
            "source_settings": {
                "name": "PeriodicDoubleGyre",
                "boundary_buffers": [0.2, 0.2],
                "x_domain": [-0, 2],
                "y_domain": [-0, 1],
                "temporal_domain": [-10, 1000],  # will be interpreted as POSIX timestamps
                "spatial_resolution": 0.05,
                "temporal_resolution": 0.05,
                "v_amplitude": A,
                "epsilon_sep": eps,
                "period_time": 2*np.pi/omega}
            },
        "forecast": None},
    "platform_dict": {
        "battery_cap_in_wh": 400.0,
        "drag_factor": 675.0,
        "dt_in_s": dt_sim,
        "motor_efficiency": 1.0,
        "solar_efficiency": 0.2,
        "solar_panel_size": 0.5,
        "u_max_in_mps": F_max,
    },
    "seaweed_dict": {"forecast": None, "hindcast": None},
    "solar_dict": {"forecast": None, "hindcast": None},
    # "spatial_boundary": {'x': [ 0, 2 ], 'y': [ 0, 1 ]},
    "use_geographic_coordinate_system": False,
    "timeout": timeout_of_simulation,
}
arena = ArenaFactory.create(scenario_config=arenaConfig)

# % Specify Navigation Problem
x_0 = PlatformState(
    lon=units.Distance(deg=init_state[0]),
    lat=units.Distance(deg=init_state[1]),
    date_time=datetime.datetime.fromtimestamp(init_state[2], tz=datetime.timezone.utc),
)
target = SpatialPoint(lon=units.Distance(deg=target_position[0]), lat=units.Distance(deg=target_position[1]))

problem = NavigationProblem(
    start_state=x_0,
    end_region=target,
    target_radius=target_radius,
    platform_dict=arenaConfig["platform_dict"],
)
# %% Debug the filter: Manual particle initialization
# draw samples distributed with some error from the 3D Hypothesis space
A_samples = [0.4, 0.3, 0.5]
eps_samples = [0.3, 0.2, 0.4]
omega_samples = [2*np.pi/1.5, 2*np.pi/1.5, 2*np.pi/1.5]

# all equally weighted initial particles then are
initial_particles = [init_state + [A, eps, omega] for A, eps, omega in zip(A_samples, eps_samples, omega_samples)]
initial_particles = np.array(initial_particles)
# %% Debug the filter: Many particles, true is not within them!
n_mc = 10_000
A_sd, eps_sd, omega_sd = 0.2, 0.2, 1.0
A_err, eps_err, omega_err = -0.2, 0.2, -0.5     # as multiples of sd

# draw samples normally distributed with some error from the 3D Hypothesis space
A_samples = np.random.normal(size=(n_mc,1))*A_sd + A + A_err*A_sd
eps_samples = np.random.normal(size=(n_mc,1))*eps_sd + eps + eps_err*eps_sd
omega_samples = np.random.normal(size=(n_mc,1))*omega_sd + omega + omega_err*omega_sd

# all equally weighted initial particles then are
initial_particles = [init_state + [A[0], eps[0], omega[0]] for A, eps, omega in zip(A_samples, eps_samples, omega_samples)]
# add true to the initial particles
initial_particles = np.array(initial_particles)
#%% Get first observation
observation = arena.reset(platform_state=problem.start_state)
# evaluate the likelihood of the particles given the observation
model = DynamicsAndObservationModel(cov_matrix=np.eye(2)*0.05, u_max=1, dt=dt_sim, random_seed=None)

# # get observations z for them p(z|s)
# print("Observations:")
# observations = model.sample_observation(states=initial_particles)
# print(observations)     # those are currents in u and v direction at that state (noisy measurements)
print("Evaluate Observation likelihoods:")
true_current_observation = np.array(observation.true_current_at_state)
likelihoods = model.evaluate_observations(states=initial_particles, observations=true_current_observation)
#% plot the particles in 2D and their size according to likelihood
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(initial_particles[:,3], initial_particles[:,5], s=likelihoods*100)
plt.xlabel('Amplitude')
plt.ylabel('omega')
plt.scatter(A, omega, s=100, c="r", label="True")
# add legend
# plt.legend()
plt.show()
# write an enumerate loop to print the likelihoods of the particles
#%% run the computation manually
point = problem.start_state.to_spatio_temporal_point()
np.array(arena.ocean_field.forecast_data_source.get_data_at_point(point))
#%% now with the model
model.currents_analytical(states=initial_particles[10].reshape(1, -1))
#%% OK there is definitely an issue! The true particle has low probability, that does not make any sense.
# Investigate it!
# Get estimated currents in u and v direction for the states/particles
particle_idx = 10
true_currents = model.currents_analytical(states=initial_particles[particle_idx].reshape(1, -1))
error = true_current_observation - true_currents
print(error)
# evaluate the probability of the observations
model.var.pdf(error)*1000
# Particle 9 is very off and has error: [[-6.52272645e-17  2.81567793e-01]] -> very likely
# Particle 10, which is the true one has error: [[-1.59954815e-17  9.71192190e-01]] -> not likely at all
# -> the pdf seems to function correctly.
# so either the analytical currents are wrong...
# I think it's because arena has different analytical dynamics now...
#%% Setting up necessary subvariables and routines
initial_particle_belief = ParticleBelief(initial_particles)
dynamics_and_observation_model = DynamicsAndObservationModel(
    cov_matrix=np.eye(2)*0.1, u_max=F_max, dt=0.1, random_seed=None)
mcts_observer = ParticleFilterObserver(initial_particle_belief, dynamics_and_observation_model, resample=False)
#%% Print weighted particles
print(mcts_observer.particle_belief_state.states[:,3:])
print(mcts_observer.particle_belief_state.weights)
#%% before anything
from ocean_navigation_simulator.controllers.pomdp_planners.visualize import plot_particles_in_2D
plot_particles_in_2D(mcts_observer.particle_belief_state, x_axis_idx=3, y_axis_idx=4, true_state=[A, eps, omega])
#%% The main simulation loop
# observation = arena.reset(platform_state=problem.start_state)
#% run it for 1 step
x_t = np.array(observation.platform_state)[:3].reshape(1, -1)  # as x, y, t numpy array
print("State x_0: ", x_t)
next_action = 0
observation = arena.step(PlatformAction.from_discrete_action(np.array([next_action])))
print("State after action: ", np.array(observation.platform_state)[:3].reshape(1, -1))
#% Update observer
mcts_observer.full_bayesian_update(next_action, np.array(observation.true_current_at_state))
#%% Print weighted particles
plot_particles_in_2D(mcts_observer.particle_belief_state, x_axis_idx=3, y_axis_idx=4, true_state=[A, eps, omega])
#%%
step = 0
max_step = 10
observation = arena.reset(platform_state=problem.start_state)

# can also run it for a fixed amount of steps
while step < max_step:

    print("==============")
    print("Iteration: ", step)
    print("Weights: ", mcts_observer.particle_belief_state.weights)

    x_t = np.array(observation.platform_state)[:3].reshape(1, -1)  # as x, y, t numpy array
    next_action = 0

    # Execute action
    observation = arena.step(PlatformAction.from_discrete_action(np.array([next_action])))
    problem_status = arena.problem_status(problem=problem)

    # Update observer
    mcts_observer.full_bayesian_update(next_action, np.array(observation.true_current_at_state))

    # Statistics if curious
    # print("Parameter estimates:")
    # print(
    #     " - Amplitude: ",
    #     np.round(np.mean(mcts_observer.particle_belief_state.states[:, 3]), 2),
    #     " +- ",
    #     np.round(np.std(mcts_observer.particle_belief_state.states[:, 3]), 2),
    # )
    # print(
    #     " - epsilon: ",
    #     np.round(np.mean(mcts_observer.particle_belief_state.states[:, 4]), 2),
    #     " +- ",
    #     np.round(np.std(mcts_observer.particle_belief_state.states[:, 4]), 2),
    # )
    # print(
    #     " - omega: ",
    #     np.round(np.mean(mcts_observer.particle_belief_state.states[:, 5]), 2),
    #     " +- ",
    #     np.round(np.std(mcts_observer.particle_belief_state.states[:, 5]), 2),
    # )

    step += 1