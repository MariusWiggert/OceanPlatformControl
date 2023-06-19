#%%
import datetime
import numpy as np
from typing import Union, List
from scipy.stats import multivariate_normal
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
from ocean_navigation_simulator.controllers.pomdp_planners.DynamicsAndObservationModels import Dyn_Obs_Model_Position_Obs
import time

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
#%% # True parameters
A = 0.4
eps = 0.3
omega = 2*np.pi/1.5 # -> this means 1.5 time-units period time
F_max = 1.0
# Start - Goal Settings
init_state = [0.3, 0.2, 0]
target_position = [1.7, 0.8]
target_radius = 0.10

dt_sim = 0.1
timeout_of_simulation = 100     # in seconds
#%
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

#%% Get initial particles
n_mc = 10_000
A_sd, eps_sd, omega_sd = 0.2, 0.2, 1.0
A_err, eps_err, omega_err = -0.2, 0.2, -0.5     # as multiples of sd

# draw samples normally distributed with some error from the 3D Hypothesis space
A_samples = np.random.normal(size=(n_mc,1))*A_sd + A + A_err*A_sd
eps_samples = np.random.normal(size=(n_mc,1))*eps_sd + eps + eps_err*eps_sd
omega_samples = np.random.normal(size=(n_mc,1))*omega_sd + omega + omega_err*omega_sd

# all equally weighted initial particles then are
initial_particles = [init_state + [A[0], eps[0], omega[0]] for A, eps, omega in zip(A_samples, eps_samples, omega_samples)]
initial_particles = np.array(initial_particles)

#%% MCTS Settings
# Setting up necessary sub-variables and routines
initial_particle_belief = ParticleBelief(initial_particles)
dynamics_and_observation_model = Dyn_Obs_Model_Position_Obs(
    cov_matrix=np.eye(2)*0.001, u_max=F_max, dt=dt_sim, random_seed=None)
# Tune: dt such that you see very different observations after 5 steps (half or full of max_depth)

# reward_function = TimeRewardFunction(target_position, target_radius)
reward_function = RewardFunction(target_position, target_radius)
rollout_policy = HeuristicPolicy(target_position)
num_planner_particles = 100
mcts_settings = {
    # 100 is very low, the higher the more flushed out the tree
    "num_mcts_simulate": 100, # number of simulate calls to MCTS (either explores children or new node) from root node.
    "max_depth": 10,    # maximum depth of the tree -> then roll-out policy
    "max_rollout_depth": 100, # how far the rollout policy goes (should be the final T)
	"rollout_subsample": 10, # for how many particles to run rollout policy (currently not parallelized)
    "rollout_style": "FO", # PO cannot be parallized (getting observations) - FO can be parallelized
    # could do 3rd rollout-style where we approximate V(b)
    "ucb_factor": 10.0, # no explore (0.1), mostly explore (100) -> depends on reward range (this with 0-1 reward per step)
    # Factors for progressive widening. How much to expand the tree.
    "dpw_k_observations": 4.0, # means sample around 4 observations per node
    "dpw_alpha_observations": 0.25, # how aggressively to expand the tree (higher means wider tree)
    # can probably reduce that as observations do not have much noise...
    "dpw_k_actions": 3.0, # means sample around 3 actions
    "dpw_alpha_actions": 0.25,
    "discount": 0.99,
    "action_space_cardinality": 8,
}

generative_particle_filter = GenerativeParticleFilter(dynamics_and_observation_model, resample=False,
                                                      no_position_uncertainty=False)
mcts_observer = ParticleFilterObserver(initial_particle_belief, dynamics_and_observation_model, resample=False)
mcts_planner = PFTDPWPlanner(generative_particle_filter, reward_function, rollout_policy,
                             rollout_value=None, specific_settings=mcts_settings)

# For viz the tree:
# - can viz the most promising action & belief for each node.
# - take the most visited. Might only be 3 depths...

#%% do a single step of MCTS
observation = arena.reset(platform_state=problem.start_state)
current_belief = mcts_observer.get_planner_particle_belief(num_planner_particles)
# run the MCTS planner for planning
next_action = mcts_planner.get_best_action(current_belief)
print("next action: ", next_action)
#%% Functions for plotting the MCTS tree
from ocean_navigation_simulator.controllers.pomdp_planners.visualize import plot_tree, plot_tree_plotly
# for small trees with labels
# plot_tree(mcts_planner.tree, node_size=1000, q_val_decimals=1, reward_decimals=1)
# for bigger trees with labels when hovering over the nodes/edges
plot_tree_plotly(mcts_planner.tree, node_size=5, q_val_decimals=1, reward_decimals=1)
#%%
from ocean_navigation_simulator.controllers.pomdp_planners.visualize import plot_belief_in_tree
plot_belief_in_tree(mcts_planner.tree, belief_state_id=100,
                    x_axis_particle_idx=4,
                    y_axis_particle_idx=5, x_axis_label="epsilon",
                    y_axis_label="omega", true_state=[A, eps, omega])
#%%
observation = arena.reset(platform_state=problem.start_state)
problem_status = 0 # 0: running, 1: success, -1: timeout, -3: out of bounds

printing_problem_status = 10
step = 0
max_step = 200


# can also run it for a fixed amount of steps
while problem_status == 0 and step < max_step:
    # Get MCTS action
    # from the outside particle filter/observer -> get a subset of particles for planning
    current_belief = mcts_observer.get_planner_particle_belief(num_planner_particles)
    planning_start = time.time()
    # run the MCTS planner for planning
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
    pos_observation = np.array(observation.platform_state)[:2].reshape(1, -1)
    mcts_observer.full_bayesian_update(action=next_action, observation=pos_observation)
    # Now set the state part of all belief states to the true state (assuming no position uncertainty)
    mcts_observer.particle_belief_state.states[:, :2] = pos_observation

    # Statistics if curious
    if step % printing_problem_status == 0:
        print("==============")
        print("Iteration: ", step)
        print("Planning Time: ", planning_time)
        print("Parameter estimates:")
        print(
            " - Amplitude: ",
            np.round(np.mean(mcts_observer.particle_belief_state.states[:,3]), 2), 
            " +- ", 
            np.round(np.std(mcts_observer.particle_belief_state.states[:,3]), 2),
        )
        print(
            " - epsilon: ",
            np.round(np.mean(mcts_observer.particle_belief_state.states[:,4]), 2), 
            " +- ", 
            np.round(np.std(mcts_observer.particle_belief_state.states[:,4]), 2),
        )
        print(
            " - omega: ",
            np.round(np.mean(mcts_observer.particle_belief_state.states[:, 5]), 2),
            " +- ",
            np.round(np.std(mcts_observer.particle_belief_state.states[:, 5]), 2),
        )
    
    step += 1

print("Simulation terminated because:", arena.problem_status_text(arena.problem_status(problem=problem)))
#%% 
# Visualize the trajectory as 2D plot
arena.plot_all_on_map(problem=problem)
#%%
from ocean_navigation_simulator.controllers.pomdp_planners.visualize import plot_particles_in_2D
plot_particles_in_2D(mcts_observer.particle_belief_state, x_axis_idx=3, y_axis_idx=4, true_state=[A, eps, omega])
#%% plot a histogram of mcts_observer.particle_belief_state
import matplotlib.pyplot as plt
plt.hist(mcts_observer.particle_belief_state.weights, bins=20, range=[0, 0.00000001])
plt.show()
#%%
idx = 3491
print(mcts_observer.particle_belief_state.states[idx, :])
#%% get the top 10 weights of mcts_observer.particle_belief_state.weights
top_10_idx = np.argsort(mcts_observer.particle_belief_state.weights)[-10:]
# print their weights
print(mcts_observer.particle_belief_state.weights[top_10_idx])
#%% 
# Render animation of the closed-loop trajectory
arena.animate_trajectory(problem=problem, output="pomdp_planner_trajectory.mp4", # this is saved under the "generated_media" folder
                         temporal_resolution=0.1)
#%% let's inspect the last tree, it should be fairly obvious!
# ≈ 100 belief nodes and 51 action nodes
# -> seems like the action nodes are expanded quite a bit with belief nodes, the q value comes from the rollout function.
# What is the real-depth? I can get that from the transition dictionary.
# But TBH, that is quite hard..
# let's use igraph to do that.
# => I need to write some viz code, otherwise not possible to understand.
#%% Functions for plotting the MCTS tree
from ocean_navigation_simulator.controllers.pomdp_planners.visualize import plot_tree, plot_tree_plotly
# for small trees with labels
# plot_tree(mcts_planner.tree, node_size=1000, q_val_decimals=1, reward_decimals=1)
# for bigger trees with labels when hovering over the nodes/edges
plot_tree_plotly(mcts_planner.tree, node_size=5, q_val_decimals=1, reward_decimals=1)
#%%
from ocean_navigation_simulator.controllers.pomdp_planners.visualize import plot_belief_in_tree
plot_belief_in_tree(mcts_planner.tree, belief_state_id=4, x_axis_particle_idx=4,
                    y_axis_particle_idx=5, x_axis_label="epsilon",
                    y_axis_label="omega", true_state=[A, eps, omega])
#%%
from ocean_navigation_simulator.controllers.pomdp_planners.visualize import plot_particles_in_2D
plot_particles_in_2D(mcts_observer.particle_belief_state, x_axis_idx=4, y_axis_idx=5, true_state=[A, eps, omega])







#%% Test the generative particle filter!
A = 0.4
eps = 0.3
omega = 2*np.pi/1.5 # -> this means 1.5 time-units period time
F_max = 1.0
# Start - Goal Settings
init_state = [0.3, 0.2, 0]
target_position = [1.7, 0.8]
target_radius = 0.10

dt_sim = 0.1
timeout_of_simulation = 100     # in seconds
A_samples = [0.4, 0.3, 0.5]
eps_samples = [0.3, 0.2, 0.4]
omega_samples = [2*np.pi/1.5, 2*np.pi/1.5, 2*np.pi/1.5]

# all equally weighted initial particles then are
initial_particles = [init_state + [A, eps, omega] for A, eps, omega in zip(A_samples, eps_samples, omega_samples)]
initial_particles = np.array(initial_particles)

# Setting up necessary sub-variables and routines
initial_particle_belief = ParticleBelief(initial_particles)
initial_particle_belief.normalize_weights()
from ocean_navigation_simulator.controllers.pomdp_planners.DynamicsAndObservationModels import Dyn_Obs_Model_Position_Obs
dynamics_and_observation_model = Dyn_Obs_Model_Position_Obs(
    cov_matrix=np.eye(2)*0.0001, u_max=F_max, dt=dt_sim, random_seed=None)

generative_particle_filter = GenerativeParticleFilter(dynamics_and_observation_model, resample=False,
                                                      no_position_uncertainty=False)
#%%
print(initial_particle_belief.weights)
#% update belief and then visualize again
next_belief = generative_particle_filter.generate_next_belief(initial_particle_belief, action=0)
print(next_belief.weights)
