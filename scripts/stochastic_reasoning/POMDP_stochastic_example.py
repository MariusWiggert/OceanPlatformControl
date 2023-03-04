#%% 
# Dynamics and Observation Model for Double Gyre Flow Planning
# Note: it is vectorized to work with as many particle in parallel as needed.
import abc
import datetime
import numpy as np
from typing import Union, List
from scipy.stats import multivariate_normal

class DynamicsAndObservationModel(abc.ABC):
    x_domain = [-0, 2]
    y_domain = [0, 1]

    def __init__(self, cov_matrix: np.array, u_max: float = 0.2, epsilon_sep: float = 0.2,
                 dt: float = 1, random_seed: int = None):
        self.random_seed = random_seed
        self.epsilon_sep = epsilon_sep
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
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
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
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
            observations: measurement vector (n, 2) with columns (u_current, v_current) as np.array
        Returns:
            p: probability of the measurement given the state (n, 1) as np.array
        """
        # Get true currents in u and v direction
        true_currents = self.currents_analytical(states=states)
        error = observations - true_currents
        # evaluate the probability of the observations
        return self.var.pdf(error)

    def get_next_states(self, states: np.array, actions: np.array) -> np.array:
        """Shallow state wrapper for dynamics model.
        Input:
            s: (n, 5) numpay array of state with columns [x, y, t, period_time, v_amplitude]
            actions: array (n,) of action integers (between 0 and 7 for directions. 0 action is 0*pi and every int after +pi/4)
        Output:
            s_next: numpy array of next states (n, 5) with columns [x, y, t, period_time, v_amplitude]
        """
        # Get true currents in u and v direction
        curs = self.currents_analytical(states=states)
        dx = (self.u_max * np.cos(actions * np.pi / 4) + curs[:, 0]) * self.dt
        dy = (self.u_max * np.sin(actions * np.pi / 4) + curs[:, 1]) * self.dt

        new_states = states + np.array([dx, dy, model.dt * np.ones(dx.shape), np.zeros(dx.shape), np.zeros(dx.shape)]).T

        return new_states

    def currents_analytical(self, states: np.array) -> np.array:
        """Analytical Formula for u and v currents of Periodic Double Gyre.
        Note: this can be used for floats as input or np.arrays which are all the same shape.
        Args:
            states: numpy array of shape (n, 5) with the following columns: [lon, lat, posix_time, period_time, v_amplitude]
        Returns:
            currents  data as numpy array (n, 2 with columns [u_current, v_current])
        """
        w_angular_vel = 2 * np.pi / states[:, 3]
        a = self.epsilon_sep * np.sin(w_angular_vel * states[:, 2])
        b = 1 - 2 * self.epsilon_sep * np.sin(w_angular_vel * states[:, 2])
        f = a * np.power(a * states[:, 0], 2) + b * states[:, 0]
        df_dx = 2 * a * states[:, 0] + b

        u_cur_out = -np.pi * states[:, 4] * np.sin(np.pi * f) * np.cos(np.pi * states[:, 1])
        v_cur_out = np.pi * states[:, 4] * np.cos(np.pi * f) * np.sin(np.pi * states[:, 1]) * df_dx
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
# Using it:
# initialize the model
model = DynamicsAndObservationModel(cov_matrix=np.eye(2), u_max=0.1, epsilon_sep=0.2, dt=0.1, random_seed=None)
# get the currents at two states (one state is: [x, y, t, period_time, v_amplitude]) states are in rows
states = np.array([[1.25, 0.5, 0, 100, 1],
                   [1.8, 0.1, 0, 75, 0.75],
                   [1.2, 0.4, 0, 50, 0.5]])
# get next states for them p(s'|s,a)
actions = np.array([0, 3, 2]) # discretized 0 - 7. 0 is 0*pi and every int after +pi/4
print("Next states:")
print(model.get_next_states(states=states, actions=actions))
# get observations z for them p(z|s)
print("Observations:")
observations = model.sample_observation(states=states)
print(observations) # those are currents in u and v direction at that state (noisy measurements)
print("Evaluate Observation likelihoods:")
print(model.evaluate_observations(states=states, observations=observations))


#%% 
# Getting an initial state distribution
platform_position = [0.1, 0.5, 0]

period_time_range = [50, 100]
v_amplitude_range = [0.5, 1.0]

# sample from this 2D space of hypothesis uniformly
n_samples = 10
period_time_samples = np.random.uniform(low=period_time_range[0], high=period_time_range[1], size=n_samples)
v_amplitude_samples = np.random.uniform(low=v_amplitude_range[0], high=v_amplitude_range[1], size=n_samples)

# all equally weighted initial particles then are
initial_particles = [platform_position + [period_time, v_amplitude] for period_time, v_amplitude in zip(period_time_samples, v_amplitude_samples)]


#%% 
# Heuristic action policy ('naive to target')
class HeuristicPolicy:
    """Heuristic policy to go to a target."""
    def __init__(self, target: np.array):
        self.target = target

    def get_action(self, states: np.array) -> np.array:
        """Heuristic policy to go to a target.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            actions: vector of actions as np.array (n,)
        """
        # get the angle to the target
        angle_to_target = np.arctan2(self.target[1] - states[:,1], self.target[0] - states[:,0])
        # discretize the angle
        actions = np.round(angle_to_target / (np.pi / 4)).astype(int)
        return actions
    

#%% 
# Get the action for a state
print("Heuristic policy:")
print(HeuristicPolicy(target=np.array([0, 0.5])).get_action(states=np.array(initial_particles)))
print("Same actions are expected because the platform position for all of them is the same, currents are ignored.")


#%% 
# Define a navigation problem to be solved
# reward function: Option 1: just negative distance to target at each point in time
class RewardFunction:
    x_domain = [-0, 2]
    y_domain = [0, 1]
    
    def __init__(self, target: np.array, target_radius: float = 0.05):
        self.target = target
        self.target_radius = target_radius

    def get_reward(self, states: np.array) -> np.array:
        """Reward function for the navigation problem.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            rewards: vector of rewards as np.array (n,)
        """
        # return the negative distance
        rewards = -1.0 * self.get_distance_to_target(states)
        rewards -= np.where(self.is_boundary(states), 100000.0, 0.0)
        return rewards
    
    def is_boundary(self, states: np.array) -> Union[float, np.array]:
        """Helper function to check if a state is in the boundary."""
        lon = states[:, 0]
        lat = states[:, 1]
        x_boundary = np.logical_or(
            lon < self.x_domain[0],
            lon > self.x_domain[1],
        )
        y_boundary = np.logical_or(
            lat < self.y_domain[0],
            lat > self.y_domain[1],
        )

        return np.logical_or(x_boundary, y_boundary)
    
    def reached_goal(self, states: np.array) -> Union[float, np.array]:
        """Helper function to check if a state reached the goal."""
        return self.get_distance_to_target(states) < 0.0
    
    def get_distance_to_target(self, states: np.array) -> Union[float, np.array]:
        """Helper function to get distance to target."""
        return np.linalg.norm(states[:, :2] - self.target, axis=1) - self.target_radius
    
    def check_terminal(self, states: np.array) -> np.array:
        """Check terminal conditions for the navigation problem.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            is_terminal: vector of boolean as np.array (n,)
        """
        return np.logical_or(self.is_boundary(states), self.reached_goal(states))
    

# Option 2: -1 when outside target and +100 when inside
class TimeRewardFunction(RewardFunction):
    def get_reward(self, states: np.array) -> np.array:
        """Reward function for the navigation problem.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            rewards: vector of rewards as np.array (n,)
        """
        # return reaching goal or terminal
        rewards = np.where(self.reached_goal(states), 100.0, 0.0)
        rewards -= np.where(self.is_boundary(states), 100000.0, 0.0)
        rewards -= 1.0
        return rewards


#%% 
# Run a simulation with the heuristic policy (Your planner only needs to implement the get_action function)
# Initialize the simulator of reality (this takes a bit, it performs caching of currents under the hood)
u_max = 0.1
dt_sim = 0.1
timeout_of_simulation = 100 # in seconds

arenaConfig = {
    "casadi_cache_dict": {"deg_around_x_t": 0.5, "time_around_x_t": 1000.0},
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
                "temporal_resolution": 10,
                "v_amplitude": 0.5,
                "epsilon_sep": 0.2,
                "period_time": 50}
            },
        "forecast": None},
    "platform_dict": {
        "battery_cap_in_wh": 400.0,
        "drag_factor": 675.0,
        "dt_in_s": dt_sim,
        "motor_efficiency": 1.0,
        "solar_efficiency": 0.2,
        "solar_panel_size": 0.5,
        "u_max_in_mps": u_max,
    },
    "seaweed_dict": {"forecast": None, "hindcast": None},
    "solar_dict": {"forecast": None, "hindcast": None},
    # "spatial_boundary": {'x': [ 0, 2 ], 'y': [ 0, 1 ]},
    "use_geographic_coordinate_system": False,
    "timeout": timeout_of_simulation,
}

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
arena = ArenaFactory.create(scenario_config=arenaConfig)


#%% 
# Working with the arena object
# visualize the true currents at a specific time
posix_time = 0
arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=datetime.datetime.fromtimestamp(posix_time, tz=datetime.timezone.utc),
    x_interval=[0, 2],
    y_interval=[0, 1],
)
# render an animation of the true currents
arena.ocean_field.hindcast_data_source.animate_data(
    x_interval=[0, 2],
    y_interval=[0, 1],
    t_interval=[0, 500],
    output="test_analytical_current_animation.mp4", # this is saved under the "generated_media" folder
)


#%% 
# This is how to run closed-loop simulations
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.Arena import PlatformAction
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import set_arena_loggers
import logging
# if you want to debug something with the closed-loop simulator (but shouldn't be necessary)
# set_arena_loggers(logging.DEBUG)

def is_inside(state: np.array) -> Union[float, np.array]:
    """Helper function to check if a state is in the boundary."""
    lon = state[0]
    lat = state[1]
    x_interval=[0, 2]
    y_interval=[0, 1]

    x_boundary = np.logical_or(
        lon < x_interval[0],
        lon > x_interval[1],
    )
    y_boundary = np.logical_or(
        lat < y_interval[0],
        lat > y_interval[1],
    )

    return np.logical_not(np.logical_or(x_boundary, y_boundary))

# set the start and target positions
platform_position = [0.1, 0.5, 0]
target_position = [1.5, 0.5]
target_radius = 0.10
reward_class = RewardFunction(target=target_position, target_radius=target_radius)

# % Specify Navigation Problem
x_0 = PlatformState(
    lon=units.Distance(deg=platform_position[0]),
    lat=units.Distance(deg=platform_position[1]),
    date_time=datetime.datetime.fromtimestamp(platform_position[2], tz=datetime.timezone.utc),
)
target = SpatialPoint(lon=units.Distance(deg=target_position[0]), lat=units.Distance(deg=target_position[1]))

problem = NavigationProblem(
    start_state=x_0,
    end_region=target,
    target_radius=target_radius,
    platform_dict=arenaConfig["platform_dict"],
)
observation = arena.reset(platform_state=problem.start_state)
# use the heuristic policy
controller = HeuristicPolicy(target=target_position)
# set initial problem status
problem_status = 0 # 0: running, 1: success, -1: timeout, -3: out of bounds


# #%% 
# # Naive Policy
# # The main simulation loop
# total_reward = 0
# # can also run it for a fixed amount of steps
# # for i in tqdm(range(100)):
# while problem_status == 0:
#     # extract action and current measurement from observation
#     x_t = np.array(observation.platform_state)[:3].reshape(1,-1) # as x, y, t numpy array
#     current_measurement = np.array(observation.true_current_at_state) # as u, v in m/s or length units per time units
#     total_reward += reward_class.get_reward(x_t)[0]
#     print(total_reward)

#     # Get action from the policy
#     discrete_action = controller.get_action(states=x_t)

#     # execute action
#     observation = arena.step(PlatformAction.from_discrete_action(discrete_action))
#     # update problem status
#     problem_status = arena.problem_status(problem=problem)

# print("Simulation terminated because:", arena.problem_status_text(arena.problem_status(problem=problem)))
# print("Final reward:", total_reward)


# #%% 
# # Visualize the trajectory as 2D plot
# arena.plot_all_on_map(problem=problem)


# #%% 
# # Render animation of the closed-loop trajectory
# arena.animate_trajectory(problem=problem, output="closed_loop_trajectory.mp4", # this is saved under the "generated_media" folder
#                          temporal_resolution=0.1)


#%% 
# MCTS Policy
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import ParticleBelief
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleFilterObserver import ParticleFilterObserver
from ocean_navigation_simulator.controllers.pomdp_planners.GenerativeParticleFilter import GenerativeParticleFilter
from ocean_navigation_simulator.controllers.pomdp_planners.PFTDPWPlanner import PFTDPWPlanner
import time

# Getting an initial state distribution
platform_position = [0.1, 0.5, 0]

period_time_range = [50, 100]
v_amplitude_range = [0.5, 1.0]

# sample from this 2D space of hypothesis uniformly
n_samples = 10_000
period_time_samples = np.random.uniform(low=period_time_range[0], high=period_time_range[1], size=n_samples)
v_amplitude_samples = np.random.uniform(low=v_amplitude_range[0], high=v_amplitude_range[1], size=n_samples)

# all equally weighted initial particles then are
initial_particles = [platform_position + [period_time, v_amplitude] for period_time, v_amplitude in zip(period_time_samples, v_amplitude_samples)]
initial_particles = np.array(initial_particles)

#%% 
# Setting up necessary subvariables and routines
initial_particle_belief = ParticleBelief(initial_particles)
dynamics_and_observation_model = DynamicsAndObservationModel(
    cov_matrix=np.eye(2), u_max=0.1, epsilon_sep=0.2, dt=0.1, random_seed=None)
reward_function = TimeRewardFunction(target_position, target_radius)
rollout_policy = HeuristicPolicy(target_position)
num_planner_particles = 100
mcts_settings = {
    "num_mcts_simulate": 100,
    "max_depth": 10,
    "max_rollout_depth": 20,
	"rollout_subsample": 10,
    "rollout_style": "FO",
    "ucb_factor": 10.0,
    "dpw_k_observations": 4.0,
    "dpw_alpha_observations": 0.25,
    "dpw_k_actions": 3.0,
    "dpw_alpha_actions": 0.25,
    "discount": 0.99,
    "action_space_cardinality": 8,
}

# Setting up particle belief and observer
generative_particle_filter = GenerativeParticleFilter(dynamics_and_observation_model, False)
mcts_observer = ParticleFilterObserver(initial_particle_belief, dynamics_and_observation_model, True)
mcts_planner = PFTDPWPlanner(generative_particle_filter, reward_function, rollout_policy, None, mcts_settings)

#%% 
# The main simulation loop
total_reward = 0
observation = arena.reset(platform_state=problem.start_state)
problem_status = 0 # 0: running, 1: success, -1: timeout, -3: out of bounds

printing_problem_status = 10
step = 0
max_step = 200


# can also run it for a fixed amount of steps
while problem_status == 0 and step < max_step:
    # Get MCTS action
    current_belief = mcts_observer.get_planner_particle_belief(num_planner_particles)
    planning_start = time.time()
    next_action = mcts_planner.get_best_action(current_belief)
    planning_end = time.time()
    planning_time = planning_end - planning_start

    # Execute action
    observation = arena.step(PlatformAction.from_discrete_action(np.array([next_action])))
    problem_status = arena.problem_status(problem=problem)

    # Manual fix for going out of bounds:
    x_t = np.array(observation.platform_state)
    if not is_inside(x_t):
        print("Early terminated because state is not inside anymore:", x_t)
        break

    # Update observer
    mcts_observer.full_bayesian_update(next_action, np.array(observation.true_current_at_state))

    # Statistics if curious
    if step % printing_problem_status == 0:
        print("==============")
        print("Iteration: ", step)
        print("Planning Time: ", planning_time)
        print("Parameter estimates:")
        print(
            " - Period Time: ", 
            np.round(np.mean(mcts_observer.particle_belief_state.states[:,3]), 2), 
            " +- ", 
            np.round(np.std(mcts_observer.particle_belief_state.states[:,3]), 2),
        )
        print(
            " - V Amplitude: ", 
            np.round(np.mean(mcts_observer.particle_belief_state.states[:,4]), 2), 
            " +- ", 
            np.round(np.std(mcts_observer.particle_belief_state.states[:,4]), 2),
        )
    
    step += 1

print("Simulation terminated because:", arena.problem_status_text(arena.problem_status(problem=problem)))
# Visualize the trajectory as 2D plot
arena.plot_all_on_map(problem=problem)


#%% 
# Render animation of the closed-loop trajectory
arena.animate_trajectory(problem=problem, output="pomdp_planner_trajectory.mp4", # this is saved under the "generated_media" folder
                         temporal_resolution=0.1)

# %%
