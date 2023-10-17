#%% This is an example of using the POMDP planner in a stochastic setting
import jax
import datetime
import numpy as np

from ocean_navigation_simulator.controllers.pomdp_planners import RolloutPolicy, Rewards
from ocean_navigation_simulator.controllers.pomdp_planners.HJRolloutValue.ParticleHJValueEstimator import \
    ParticleHJValueEstimator
from ocean_navigation_simulator.controllers.pomdp_planners.PomdpPlanner import PomdpPlanner
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.Arena import PlatformAction
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.controllers.pomdp_planners import visualize
#% Problem Setup
# Start - Goal Settings
init_state = [5., 2.0, 0]
target_state = [5., 8.0]
target_radius = 0.5

# Ground truth simulation setup
u_highway_true = 0.5
F_max = 0.4
dt_sim = 0.1
timeout_of_simulation = 20     # in seconds

# the ensembles/particles
u_highway_samples = np.linspace(-0.5, 0.5, 3).reshape(-1, 1)
# u_highway_samples = np.array([0.5]).reshape(1, 1)

# about the observations
dt_obs = 0.5
obs_noise = 1e-3#0.005

# TODO: need to change that in analytical function it uses casadi (maybe in the arena it does it alread?)

#% instantiate true environment
arenaConfig = {
    "casadi_cache_dict": {"deg_around_x_t": 5.0,
                          "time_around_x_t": 100.0},
    "ocean_dict": {
        "hindcast": {
                "field": "OceanCurrents",
                "source": "analytical",
                "source_settings": {
                    "name": "FixedCurrentHighway",
                    "boundary_buffers": [0.2, 0.2],
                    "x_domain": [0, 10],
                    "y_domain": [0, 10],
                    "temporal_domain": [0, 100],
                    "spatial_resolution": 0.05,
                    "temporal_resolution": 1.0,
                    "y_range_highway": [4, 6],
                    "U_cur": u_highway_true,
                },
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
true_arena = ArenaFactory.create(scenario_config=arenaConfig)
# % Specify Navigation Problem
x_0 = PlatformState(
    lon=units.Distance(deg=init_state[0]),
    lat=units.Distance(deg=init_state[1]),
    date_time=datetime.datetime.fromtimestamp(init_state[2], tz=datetime.timezone.utc),
)
target = SpatialPoint(lon=units.Distance(deg=target_state[0]), lat=units.Distance(deg=target_state[1]))

problem = NavigationProblem(
    start_state=x_0,
    end_region=target,
    target_radius=target_radius,
    platform_dict=arenaConfig["platform_dict"],
    x_range=[0, 10],
    y_range=[0, 10],
)
#% Set up the value function estimator
import jax.numpy as jnp

specific_settings = {
        "n_time_vector": 200,
        "T_goal_in_units": 20,
        "discrete_actions": True,
        "grid_res": 0.05,
        "ttr_to_rewards": lambda x: -(x * 10)  # + 100
}
def highway_vel(t, x, y, u_highway, y_range_highway):
    u_cur_out = jnp.where(jnp.logical_and(y_range_highway[0] <= y, y <= y_range_highway[1]), u_highway, 0.0)
    # adding 0 in the y direction
    return jnp.array([u_cur_out, 0.0])

# use functools partial to fix y_range_highway to [2,5]
from functools import partial
highway_vel_fixed = partial(highway_vel, y_range_highway=[4,6])

value_estimator = ParticleHJValueEstimator(problem, specific_settings, vel_func=jax.jit(highway_vel_fixed))
#% #% Now we can instantite the pomdp planner
_key = jax.random.PRNGKey(10)
#% Settings for the outside particle filter
particle_filter_dict = {
    'dt_observations': dt_obs,
    'resample': False,
    'no_position_uncertainty': True,
    'n_euler_per_dt_dynamics': 50,
}
# Settings for inside
mcts_dict = {
    'num_planner_particles': 3, # manually set right now...
    'dt_mcts': 0.5,
    'n_euler_per_dt_dynamics': 10,
    'n_states': 2,
    'n_actions': 9,
    'rollout_policy': None, #RolloutPolicy.NaiveToTarget(target_state),
    'rollout_value': value_estimator,
    'no_position_uncertainty': True,
    'resample': False,
    'reward_function': Rewards.TimeRewardFunction(target_state, target_radius), # Rewards.RewardFunction(target_state, target_radius),
    'mcts_settings': {
        # 100 is very low, the higher the more flushed out the tree
        "num_mcts_simulate": 200,  # number of simulate calls to MCTS (either explores children or new node) from root node.
        "max_depth": 1,  # maximum depth of the tree -> then roll-out policy
        "max_rollout_depth": 200,  # how far the rollout policy goes (should be the final T)
        "rollout_subsample": 10,  # for how many particles to run rollout policy (currently not parallelized)
        "rollout_style": "FO",  # PO cannot be parallized (getting observations) - FO can be parallelized
        # could do 3rd rollout-style where we approximate V(b)
        "ucb_factor": 10.0,
        # no explore (0.1), mostly explore (100) -> depends on reward range (this with 0-1 reward per step)
        # Factors for progressive widening. How much to expand the tree.
        "dpw_k_observations": 20.,  # means sample around 4 observations per node
        "dpw_alpha_observations": 0.1,  # how aggressively to expand the tree (higher means wider tree)
        # can probably reduce that as observations do not have much noise...
        "dpw_k_actions": 5.0,  # means sample around 3 actions
        "dpw_alpha_actions": 0.25,
        "discount": 0.99,
        "action_space_cardinality": 9,
    }
}

from ocean_navigation_simulator.controllers.pomdp_planners.utilities.analytical_currents import highway_current_analytical

particle_vel_func = partial(highway_current_analytical,
                            y_range_highway=arenaConfig['ocean_dict']['hindcast']['source_settings']['y_range_highway'])

#%% start in the middle, otherwise no signal in the observation
init_filter_test = [5, 5, 0]
x_0_middle_of_highway = PlatformState(
    lon=units.Distance(deg=init_filter_test[0]),
    lat=units.Distance(deg=init_filter_test[1]),
    date_time=datetime.datetime.fromtimestamp(init_state[2], tz=datetime.timezone.utc),
)
pomdop_planner_for_filter_tests = PomdpPlanner(
    x_target=target_state,
    particle_vel_func=particle_vel_func,
    stoch_params=u_highway_samples,
    F_max=F_max,
    t_init=init_filter_test[2], init_state=init_filter_test[:2],
    obs_noise=obs_noise,
    key=_key,
    particle_filter_dict=particle_filter_dict,
    mcts_dict=mcts_dict)

#%% Test 0: visualize belief state at the start
visualize.plot_particles_in_1D(
    particle_belief=pomdop_planner_for_filter_tests.particle_observer.particle_belief_state,
     particle_axis_idx=3,
     particle_axis_label="u_highway",
     true_state=u_highway_true)

#%% Test 2: run just the data assimilation outer loop and inspect convergence to true

# set it up
fixed_action = PlatformAction.from_discrete_action(2)   # no action is 8
observation = true_arena.reset(platform_state=x_0_middle_of_highway)
for _ in range(int(dt_obs/dt_sim)):
    observation = true_arena.step(fixed_action)

# run assimilation step
pomdop_planner_for_filter_tests.assimilate(
    x_obs=np.array(observation.platform_state.to_spatial_point()),
    action=fixed_action.to_discrete_action())

visualize.plot_particles_in_1D(
    particle_belief=pomdop_planner_for_filter_tests.particle_observer.particle_belief_state,
     particle_axis_idx=3,
     particle_axis_label="u_highway",
     true_state=u_highway_true)

# print particle weights
print(pomdop_planner_for_filter_tests.particle_observer.particle_belief_state.weights)
# => it essentially converges after 1 step for now...
#%% Now instantiate the POMDP Planner for actually running MCTS
pomdop_planner = PomdpPlanner(
    x_target=target_state,
    particle_vel_func=particle_vel_func,
    stoch_params=u_highway_samples,
    F_max=F_max,
    t_init=init_state[2], init_state=init_state[:2],
    obs_noise=obs_noise,
    key=_key,
    particle_filter_dict=particle_filter_dict,
    mcts_dict=mcts_dict)

#%% Test 3: Run the tree once and inspect it
import time
start = time.time()
action = pomdop_planner.get_action(observation=observation)
end = time.time()
print("action", PlatformAction.to_discrete_action(action))
print(end - start)
# visualize tree
visualize.plot_tree_plotly(pomdop_planner.mcts_planner.tree, node_size=5, q_val_decimals=1, reward_decimals=1)

# Observation: the optimal action is the 2 action (straight ahead).
# that is because it's hedging against the uncertainty.
#%% Note we can inspectx the belief states at specific belief_ids
belief_state_id = 200
visualize.plot_belief_in_tree_1D(pomdop_planner.mcts_planner.tree, belief_state_id, true_state=u_highway_true)
# Note: they all look the same, no data assimilation during MCTS planning.
# That makes sense because it only plans a few steps ahead and doesn't reach the current yet.

#%% Test 4: run POMDP planner closed-loop
observation = true_arena.reset(platform_state=x_0)
problem_status = true_arena.problem_status(problem=problem)
step = 0
while problem_status == 0:
    # Get action
    action = pomdop_planner.get_action(observation=observation)
    print("step: ", step,
          "\nplanner_state:", pomdop_planner.mcts_planner.tree.belief_id_to_belief[0].states[0,:3],
          "\nparameters", pomdop_planner.mcts_planner.tree.belief_id_to_belief[0].states[:, 3],
          "\nweights", pomdop_planner.mcts_planner.tree.belief_id_to_belief[0].weights,
          "\naction: ", action.to_discrete_action())
    # observation = true_arena.step(action)
    for _ in range(int(dt_obs / dt_sim)):
        observation = true_arena.step(action)
    print("next Observation: ", observation.platform_state)
    pomdop_planner.assimilate(x_obs=np.array(observation.platform_state)[:2], action=action.to_discrete_action())
    step += 1
    # update problem status
    problem_status = true_arena.problem_status(problem=problem)

true_arena.plot_all_on_map(problem=problem)

#% compute reward
state_traj = true_arena.state_trajectory[:, :3]
reward_func_time = Rewards.TimeRewardFunction(target_state, target_radius)
reward_time = reward_func_time.get_reward(state_traj)
print(f"Reward time: {reward_time.sum() + 100}")
# Note: Optimal action is  straight (hedging) until first observation.
# then outside particle filter is converged and does the optimal action
# using the weighted HJ Value function.
#%%
print(pomdop_planner.particle_observer.particle_belief_state.states)
print(pomdop_planner.particle_observer.particle_belief_state.weights)

#%% Manual continue
# inspect the belief state!
visualize.plot_particles_in_1D(
    particle_belief=pomdop_planner.particle_observer.particle_belief_state,
     particle_axis_idx=3,
     particle_axis_label="u_highway",
     true_state=u_highway_true)

#%% To debug this. Let's fix the position and run the tree with different settings.
visualize.plot_tree_plotly(pomdop_planner.mcts_planner.tree, node_size=5, q_val_decimals=1, reward_decimals=1)
#%% asimilate
for _ in range(int(dt_obs / dt_sim)):
    observation = true_arena.step(action)
print("next Observation: ", observation.platform_state)
# pomdop_planner.assimilate(x_obs=np.array(observation.platform_state)[:2], action=action.to_discrete_action())

#%%
mcts_dict = {
    'num_planner_particles': 3, # manually set right now...
    'dt_mcts': 0.3,
    'n_euler_per_dt_dynamics': 10,
    'n_states': 2,
    'n_actions': 9,
    'rollout_policy': None, #RolloutPolicy.NaiveToTarget(target_state),
    'rollout_value': value_estimator,
    'no_position_uncertainty': True,
    'resample': False,
    'reward_function': Rewards.TimeRewardFunction(target_state, target_radius), # Rewards.RewardFunction(target_state, target_radius),
    'mcts_settings': {
        # 100 is very low, the higher the more flushed out the tree
        "num_mcts_simulate": 10,  # number of simulate calls to MCTS (either explores children or new node) from root node.
        "max_depth": 5,  # maximum depth of the tree -> then roll-out policy
        "max_rollout_depth": 200,  # how far the rollout policy goes (should be the final T)
        "rollout_subsample": 10,  # for how many particles to run rollout policy (currently not parallelized)
        "rollout_style": "FO",  # PO cannot be parallized (getting observations) - FO can be parallelized
        # could do 3rd rollout-style where we approximate V(b)
        "ucb_factor": 10.0,
        # no explore (0.1), mostly explore (100) -> depends on reward range (this with 0-1 reward per step)
        # Factors for progressive widening. How much to expand the tree.
        "dpw_k_observations": 20.,  # means sample around 4 observations per node
        "dpw_alpha_observations": 0.1,  # how aggressively to expand the tree (higher means wider tree)
        # can probably reduce that as observations do not have much noise...
        "dpw_k_actions": 5.0,  # means sample around 3 actions
        "dpw_alpha_actions": 0.25,
        "discount": 0.99,
        "action_space_cardinality": 9,
    }
}
pomdop_planner = PomdpPlanner(
    x_target=target_state,
    particle_vel_func=particle_vel_func,
    stoch_params=u_highway_samples,
    F_max=F_max,
    t_init=init_state[2], init_state=init_state[:2],
    obs_noise=obs_noise,
    key=_key,
    particle_filter_dict=particle_filter_dict,
    mcts_dict=mcts_dict)
observation = true_arena.reset(platform_state=x_0)
problem_status = true_arena.problem_status(problem=problem)
step = 0
while problem_status == 0:
    # Get action
    action = pomdop_planner.get_action(observation=observation)
    print("step: ", step, "\nplanner_state:", pomdop_planner.mcts_planner.tree.belief_id_to_belief[0].states, "action: ", action.to_discrete_action())
    if action.to_discrete_action() != 2:
        break
    for _ in range(int(dt_obs / dt_sim)):
        observation = true_arena.step(action)

#%% To debug this. Let's fix the position and run the tree with different settings.
init_state = np.array([5, 4, 5])
x_start_highway = PlatformState(
    lon=units.Distance(deg=init_state[0]),
    lat=units.Distance(deg=init_state[1]),
    date_time=datetime.datetime.fromtimestamp(init_state[2], tz=datetime.timezone.utc),
)
observation = true_arena.reset(platform_state=x_start_highway)
mcts_dict = {
    'num_planner_particles': 3, # manually set right now...
    'dt_mcts': 1.0,
    'n_euler_per_dt_dynamics': 10,
    'n_states': 2,
    'n_actions': 9,
    'rollout_policy': None, #RolloutPolicy.NaiveToTarget(target_state),
    'rollout_value': value_estimator,
    'no_position_uncertainty': True,
    'resample': False,
    'reward_function': Rewards.TimeRewardFunction(target_state, target_radius), # Rewards.RewardFunction(target_state, target_radius),
    'mcts_settings': {
        # 100 is very low, the higher the more flushed out the tree
        "num_mcts_simulate": 100,  # number of simulate calls to MCTS (either explores children or new node) from root node.
        "max_depth": 1,  # maximum depth of the tree -> then roll-out policy
        "max_rollout_depth": 200,  # how far the rollout policy goes (should be the final T)
        "rollout_subsample": 10,  # for how many particles to run rollout policy (currently not parallelized)
        "rollout_style": "FO",  # PO cannot be parallized (getting observations) - FO can be parallelized
        # could do 3rd rollout-style where we approximate V(b)
        "ucb_factor": 10.0,
        # no explore (0.1), mostly explore (100) -> depends on reward range (this with 0-1 reward per step)
        # Factors for progressive widening. How much to expand the tree.
        "dpw_k_observations": 20.,  # means sample around 4 observations per node
        "dpw_alpha_observations": 0.1,  # how aggressively to expand the tree (higher means wider tree)
        # can probably reduce that as observations do not have much noise...
        "dpw_k_actions": 5.0,  # means sample around 3 actions
        "dpw_alpha_actions": 0.25,
        "discount": 0.99,
        "action_space_cardinality": 9,
    }
}
pomdop_planner = PomdpPlanner(
    x_target=target_state,
    particle_vel_func=particle_vel_func,
    stoch_params=u_highway_samples,
    F_max=F_max,
    t_init=init_state[2], init_state=init_state[:2],
    obs_noise=obs_noise,
    key=_key,
    particle_filter_dict=particle_filter_dict,
    mcts_dict=mcts_dict)
#%%
action = pomdop_planner.get_action(observation=observation)
print("action", action.to_discrete_action())
#%% run tree
actions_taken = []
for i in range(100):
    action = pomdop_planner.get_action(observation=observation)
    print("action", action.to_discrete_action())
    actions_taken.append(action.to_discrete_action())
#% make a historgram with actions taken (they are discrete numbers 0-8)
import matplotlib.pyplot as plt
actions_taken = np.array(actions_taken)
# plt.hist(actions_taken)
bins = np.arange(0, 10) - 0.5
# then you plot away
fig, ax = plt.subplots()
_ = ax.hist(actions_taken, bins)
ax.set_xticks(bins + 0.5)
plt.show()
#%%
visualize.plot_tree_plotly(pomdop_planner.mcts_planner.tree, node_size=5, q_val_decimals=1, reward_decimals=1)
#%% Let's look at belief ID 5. After Action 4.
belief_state_id = 15
visualize.plot_belief_in_tree_1D(pomdop_planner.mcts_planner.tree, belief_state_id, true_state=u_highway_true)
# What happened is that it sampled observation from u_highway=0 so collapse to 0 and evaluate based on that.
#%% why not action 2?
belief_state_id = 7
visualize.plot_belief_in_tree_1D(pomdop_planner.mcts_planner.tree, belief_state_id, true_state=u_highway_true)
# it samples observation from u_highway=0.5 -> true and it knows the expected TTR then but it's not great because detour.
#%% check rollout_value from belief 5
belief_state_id = 7
value_estimator(pomdop_planner.mcts_planner.tree.belief_id_to_belief[belief_state_id])
#%% visualize both TTR functions
value_estimator.plot_ttr_snapshot(stoch_params_idx=0, time_idx=-1, granularity=2, time_to_reach=True)
#%% TLDR: the algorithm does what it should be doing... more exploration/compute should fix it.


