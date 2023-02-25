#%% Dynamics and Observation Model for Double Gyre Flow Planning
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

        return np.logical_or(x_boundary, y_boundary)
#% Using it:
# initialize the model
model = DynamicsAndObservationModel(cov_matrix=np.eye(2), u_max=0.1, epsilon_sep=0.2, dt=0.1, random_seed=None)
# get the currents at two states (one state is: [x, y, t, period_time, v_amplitude]) states are in rows
states = np.array([[1.25, 0.5, 0, 100, 1],
                   [1.8, 0.1, 0, 100, 1]])
# get next states for them p(s'|s,a)
actions = np.array([0, 3]) # discretized 0 - 7. 0 is 0*pi and every int after +pi/4
print("Next states:")
print(model.get_next_states(states=states, actions=actions))
# get observations z for them p(z|s)
print("Observations:")
observations = model.sample_observation(states=states)
print(observations) # those are currents in u and v direction at that state (noisy measurements)
print("Evaluate Observation likelihoods:")
print(model.evaluate_observations(states=states, observations=observations))

#% Getting an initial state distribution
platform_position = [0.1, 0.5, 0]

period_time_range = [50, 100]
v_amplitude_range = [0.5, 1.0]

# sample from this 2D space of hypothesis uniformly
n_samples = 10
period_time_samples = np.random.uniform(low=period_time_range[0], high=period_time_range[1], size=n_samples)
v_amplitude_samples = np.random.uniform(low=v_amplitude_range[0], high=v_amplitude_range[1], size=n_samples)

# all equally weighted initial particles then are
initial_particles = [platform_position + [period_time, v_amplitude] for period_time, v_amplitude in zip(period_time_samples, v_amplitude_samples)]

#% heuristic action policy ('naive to target')
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
#% Get the action for a state
print("Heuristic policy:")
print(HeuristicPolicy(target=np.array([0, 0.5])).get_action(states=np.array(initial_particles)))
print("Same actions are expected because the platform position for all of them is the same, currents are ignored.")

#% define a navigation problem to be solved
# reward function: Option 1: just negative distance to target at each point in time
class RewardFunction:
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
        # get the distance to the target
        distance_to_target = np.linalg.norm(states[:, :2] - self.target, axis=1) - self.target_radius
        # return the negative distance
        return -distance_to_target
# Option 2: -1 when outside target and +100 when inside
class TimeRewardFunction(RewardFunction):
    def get_reward(self, states: np.array) -> np.array:
        """Reward function for the navigation problem.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            rewards: vector of rewards as np.array (n,)
        """
        # return the negative distance
        return -1 + np.where(np.linalg.norm(states[:, :2] - self.target, axis=1) < self.target_radius, 100, 0)

#% Run a simulation with the heuristic policy (Your planner only needs to implement the get_action function)
# Initialize the simulator of reality (this takes a bit, it performs caching of currents under the hood)
u_max = 0.1
dt_sim = 0.03
timeout_of_simulation = 100 # in seconds

arenaConfig = {
    "casadi_cache_dict": {"deg_around_x_t": 2., "time_around_x_t": 50.0},
    "ocean_dict": {
        "hindcast": {
            "field": "OceanCurrents",
            "source": "analytical",
            "source_settings": {
                "name": "PeriodicDoubleGyre",
                "boundary_buffers": [0.0, 0.0],
                "x_domain": [0, 2],
                "y_domain": [0, 1.],
                "temporal_domain": [0, 100],  # will be interpreted as POSIX timestamps
                "spatial_resolution": 0.02,
                "temporal_resolution": 0.05,
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
#% Working with the arena object
# visualize the true currents at a specific time
# posix_time = 0
# arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=datetime.datetime.fromtimestamp(posix_time, tz=datetime.timezone.utc),
#     x_interval=[-0.4, 2.4],
#     y_interval=[-0.2, 1.2],
# )
# render an animation of the true currents
# arena.ocean_field.hindcast_data_source.animate_data(
#     x_interval=[0, 2],
#     y_interval=[0, 1],
#     t_interval=[0, 500],
#     output="test_analytical_current_animation.mp4", # this is saved under the "generated_media" folder
# )
#% This is how to run closed-loop simulations
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.Arena import PlatformAction
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import set_arena_loggers
import logging
# if you want to debug something with the closed-loop simulator (but shouldn't be necessary)
# set_arena_loggers(logging.DEBUG)

# set the start and target positions
platform_position = [0.1, 0.5, 0]
target_position = [1.5, 0.5]
target_radius = 0.05
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
problem_status = 0  # 0: running, 1: success, -1: timeout, -3: out of bounds
#%% The main simulation loop
total_reward = 0
# can also run it for a fixed amount of steps
# for i in tqdm(range(100)):
while problem_status == 0:
    # extract action and current measurement from observation
    x_t = np.array(observation.platform_state)[:3].reshape(1,-1) # as x, y, t numpy array
    current_measurement = np.array(observation.true_current_at_state) # as u, v in m/s or length units per time units
    total_reward += reward_class.get_reward(x_t)[0]
    print(total_reward)

    # Get action from the policy
    discrete_action = controller.get_action(states=x_t)

    # execute action
    observation = arena.step(PlatformAction.from_discrete_action(discrete_action))
    # update problem status
    problem_status = arena.problem_status(problem=problem)

print("Simulation terminated because:", arena.problem_status_text(arena.problem_status(problem=problem)))
print("Final reward:", total_reward)
# With Heuristic Policy: -839 (a lot of detours) after 40.6 seconds
# with best in hindsight for the Heuristic Controller
#%% visualize the trajectory as 2D plot
arena.plot_all_on_map(problem=problem)
#%% render animation of the closed-loop trajectory
arena.animate_trajectory(problem=problem, output="closed_loop_trajectory.mp4", # this is saved under the "generated_media" folder
                         temporal_resolution=0.1)

#%% The true time-optimal controller using reachability value function closed loop
observation = arena.reset(platform_state=problem.start_state)
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
specific_settings = {
    "replan_on_new_fmrc": False,
    "direction": "multi-time-reach-back",
    "n_time_vector": 200,
    "closed_loop": True,  # to run closed-loop or open-loop
    "T_goal_in_seconds": 10, # this led to issues because the temporal resolution of the source was 1!
    "use_geographic_coordinate_system": False,
    "progress_bar": True,
    "deg_around_xt_xT_box": 2.0,
    "grid_res": 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
}
hj_controller = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
#% run hj to compute the value function on the true currents
_ = hj_controller.get_action(observation=observation)
#%% vizualize the value function
import hj_reachability as hj
hj.viz.visFunc(hj_controller.grid, hj_controller.all_values[0,...])
#%%
hj_controller.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=1,
    alpha_color=1,
    time_to_reach=False,
    fig_size_inches=(12, 12),
    plot_in_h=False,
)
#%%
hj_controller.grid.boundary_conditions
#%%
hj_controller.plot_reachability_animation(time_to_reach=False,
                                          plot_in_h=False,
                                          temporal_resolution=0.1,
                                          with_background=True,
                                          filename="test_reach_animation.mp4")
#%% run closed-loop simulation
problem_status = 0
total_reward = 0
from tqdm import tqdm
for i in tqdm(range(300)):
    if problem_status != 0:
        break
# while problem_status == 0:
    # extract action and current measurement from observation
    x_t = np.array(observation.platform_state)[:3].reshape(1,-1) # as x, y, t numpy array
    total_reward += reward_class.get_reward(x_t)[0]

    # Get action from the policy
    action = hj_controller.get_action(observation=observation)

    # execute action
    observation = arena.step(action)
    # update problem status
    problem_status = arena.problem_status(problem=problem)

print("Simulation terminated because:", arena.problem_status_text(arena.problem_status(problem=problem)))
print("Final reward:", total_reward)
# final reward is: -330... but not compareable to above yet.
#%%
arena.plot_all_on_map(problem=problem)
#%%
arena.animate_trajectory(problem=problem, output="closed_loop_trajectory.mp4", # this is saved under the "generated_media" folder
                         temporal_resolution=0.1)