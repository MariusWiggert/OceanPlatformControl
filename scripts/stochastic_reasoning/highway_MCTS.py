#%%
from copy import deepcopy
from functools import partial
import jax
import datetime
import numpy as np
from tqdm import tqdm

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.controllers.pomdp_planners import RolloutPolicy, Rewards
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
init_state = [5., 2, 0]
target_state = [5., 8.0]
target_radius = 0.5

# Ground truth simulation setup
u_highway = 0.5
F_max = 0.4
dt_sim = 0.1
timeout_of_simulation = 20     # in seconds

# about the observations -> THIS IS KEY!
dt_obs = 0.1
obs_noise = 1e-3#0.005

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
                    "spatial_resolution": 0.1,
                    "temporal_resolution": 1.0,
                    "y_range_highway": [4, 6],
                    "U_cur": u_highway,
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
arena = ArenaFactory.create(scenario_config=arenaConfig)
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
)

# #%% inspect the currents
# # visualize the true currents at a specific time
# posix_time = 0
# arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=datetime.datetime.fromtimestamp(posix_time, tz=datetime.timezone.utc),
#     x_interval=arenaConfig['ocean_dict']['hindcast']['source_settings']['x_domain'],
#     y_interval=arenaConfig['ocean_dict']['hindcast']['source_settings']['y_domain'],
# )
#% Get true optimal control
specific_settings = {
    "replan_on_new_fmrc": False,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",
    "n_time_vector": 200,
    "closed_loop": True,  # to run closed-loop or open-loop
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 4.0,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 20,
    "use_geographic_coordinate_system": False,
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.05,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "platform_dict": arena.platform.platform_dict,
}
from ocean_navigation_simulator.controllers.NaiveController import NaiveController
# hj_planner = NaiveController(problem=problem, specific_settings=specific_settings)
hj_planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
# % Run reachability planner
observation = arena.reset(platform_state=x_0)
action = hj_planner.get_action(observation=observation)
#%%
hj_planner.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=False,
)
# # hj_planner.plot_reachability_snapshot_over_currents(rel_time_in_seconds=0, time_to_reach=True)
#%% HJ discretized
observation = arena.reset(platform_state=x_0)
problem_status = arena.problem_status(problem=problem)
step = 0
while problem_status == 0:
    # Get action
    action = hj_planner.get_action(observation=observation)
    # # modify action to discrete action
    # disc_action = action.to_discrete_action()
    # # now to action again
    # action = PlatformAction.from_discrete_action(disc_action)
    observation = arena.step(action)
    step += 1
    # update problem status
    problem_status = arena.problem_status(problem=problem)

arena.plot_all_on_map(problem=problem)
#%%
#% compute reward
state_traj = arena.state_trajectory[:, :3]
reward_func_time = Rewards.TimeRewardFunction(target_state, target_radius)
reward_time = reward_func_time.get_reward(state_traj)
print(f"Reward time: {reward_time.sum()}")
reward_func_dist = Rewards.RewardFunction(target_state, target_radius)
reward_dist = reward_func_dist.get_reward(state_traj)
print(f"Reward dist: {reward_dist.sum()}")

#%% Naive discretized
planner = RolloutPolicy.NaiveToTarget(target_state)
observation = arena.reset(platform_state=x_0)
problem_status = arena.problem_status(problem=problem)
step = 0
while problem_status == 0:
    # Get action
    discrete_action = planner.get_action(states=np.array(observation.platform_state).reshape(1, -1))
    action = PlatformAction.from_discrete_action(discrete_action[0])
    observation = arena.step(action)
    step += 1
    # update problem status
    problem_status = arena.problem_status(problem=problem)

arena.plot_all_on_map(problem=problem)
#%
#% compute reward
state_traj = arena.state_trajectory[:, :3]
reward_func_time = Rewards.TimeRewardFunction(target_state, target_radius)
reward_time = reward_func_time.get_reward(state_traj)
print(f"Reward time: {reward_time.sum()}")
reward_func_dist = Rewards.RewardFunction(target_state, target_radius)
reward_dist = reward_func_dist.get_reward(state_traj)
print(f"Reward dist: {reward_dist.sum()}")

#%%
observation = arena.reset(platform_state=x_0)
problem_status = arena.problem_status(problem=problem)
step = 0
while problem_status == 0:
    # Get action
    action = planner.get_action(observation=observation)
    print("step: ", step, "planner_state:", planner.mcts_planner.tree.belief_id_to_belief[0].states, "action: ", action.to_discrete_action())
    observation = arena.step(action)
    print("next Observation: ", observation.platform_state)
    planner.assimilate(x_obs=np.array(observation.platform_state)[:2], action=action.to_discrete_action())
    step += 1
    # update problem status
    problem_status = arena.problem_status(problem=problem)

arena.plot_all_on_map(problem=problem)

#% compute reward
state_traj = arena.state_trajectory[:, :3]
reward_func_time = Rewards.TimeRewardFunction(target_state, target_radius)
reward_time = reward_func_time.get_reward(state_traj)
print(f"Reward time: {reward_time.sum() + 100}")
reward_func_dist = Rewards.RewardFunction(target_state, target_radius)
reward_dist = reward_func_dist.get_reward(state_traj)
print(f"Reward dist: {reward_dist.sum()}")
#%% for discretized naive...
planner = RolloutPolicy.NaiveToTarget(target_state)
discrete_action = planner.get_action(states= np.array(observation.platform_state))
action = PlatformAction.from_discrete_action(discrete_action)
#%%
from ocean_navigation_simulator.controllers.pomdp_planners.RolloutValue import get_value_from_hj
from functools import partial

get_value_from_hj_partial = partial(get_value_from_hj, hj_planner=hj_planner)

#% Get initial particles
n_mc = 100
u_highway_sd = 0.5
u_highway_err = 0   # -0.2   # as multiples of sd
import jax.numpy as jnp
_key = jax.random.PRNGKey(2)
_key = jnp.array([3,3], dtype=jnp.uint32)
print(_key)
# draw samples normally distributed with some error from the 3D Hypothesis space
u_highway_samples = jax.random.normal(_key, shape=(n_mc,1))*u_highway_sd + u_highway + u_highway_err*u_highway_sd
# transform jax.numpy array to numpy array
u_highway_samples = np.array(u_highway_samples)
u_highway_samples = np.linspace(-0.5, 0.5, 5).reshape(-1,1)
u_highway_samples = np.array([0.5]).reshape(-1,1)
#% Settings for the
particle_filter_dict = {
    'dt_observations': dt_obs,
    'resample': False,
    'no_position_uncertainty': True,
    'n_euler_per_dt_dynamics': 50,
}
# Idea from claire: Can we make MCTS step-size adaptive by if it's different then what we expect?
mcts_dict = {
    'num_planner_particles': 1,
    'dt_mcts': 0.1,
    'n_euler_per_dt_dynamics': 1,
    'n_states': 2,
    'n_actions': 9,
    'rollout_policy': None, #RolloutPolicy.NaiveToTarget(target_state),
    'rollout_value': get_value_from_hj_partial,
    'no_position_uncertainty': False,
    'resample': False,
    'reward_function': Rewards.TimeRewardFunction(target_state, target_radius), # Rewards.RewardFunction(target_state, target_radius),
    'mcts_settings': {
        # 100 is very low, the higher the more flushed out the tree
        "num_mcts_simulate": 500,  # number of simulate calls to MCTS (either explores children or new node) from root node.
        "max_depth": 10,  # maximum depth of the tree -> then roll-out policy
        "max_rollout_depth": 200,  # how far the rollout policy goes (should be the final T)
        "rollout_subsample": 10,  # for how many particles to run rollout policy (currently not parallelized)
        "rollout_style": "FO",  # PO cannot be parallized (getting observations) - FO can be parallelized
        # could do 3rd rollout-style where we approximate V(b)
        "ucb_factor": 10.0,
        # no explore (0.1), mostly explore (100) -> depends on reward range (this with 0-1 reward per step)
        # Factors for progressive widening. How much to expand the tree.
        "dpw_k_observations": 1.0,  # means sample around 4 observations per node
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
planner = PomdpPlanner(
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
# check if basics work
visualize.plot_particles_in_1D(
    particle_belief=planner.particle_observer.particle_belief_state,
     particle_axis_idx=3,
     particle_axis_label="u_highway",
     true_state=u_highway)
#%% one step
action = PlatformAction.from_discrete_action(2)   # no action is 8
x_0.lon = units.Distance(deg=5)
observation = arena.reset(platform_state=x_0)
print(observation)
for _ in range(int(dt_obs/dt_sim)):
    observation = arena.step(action)
print(observation)
planner.assimilate(x_obs=np.array(observation.platform_state.to_spatial_point()),
                   action=action.to_discrete_action())
visualize.plot_particles_in_1D(
    particle_belief=planner.particle_observer.particle_belief_state,
     particle_axis_idx=3,
     particle_axis_label="u_highway",
     true_state=u_highway)
#%%
planner.particle_observer.particle_belief_state.weights
#%%
# measure how long this takes
# x_0.lon = units.Distance(deg=6)
# x_0.lat = units.Distance(deg=9)
# observation = arena.reset(platform_state=x_0)
import time
start = time.time()
action = planner.get_action(observation=observation)
end = time.time()
print("action", PlatformAction.to_discrete_action(action))
print(end - start)
# 600s -> now 4s with more euler steps... before with 1 euler step it was 1s.
# can reduce that again with jax.jit and scan I belief.

#%% for small trees with labels
# visualize.plot_tree(mcts_planner.tree, node_size=1000, q_val_decimals=1, reward_decimals=1)
# for bigger trees with labels when hovering over the nodes/edges
visualize.plot_tree_plotly(planner.mcts_planner.tree, node_size=5, q_val_decimals=1, reward_decimals=1)
#%%
planner.mcts_planner.tree.belief_id_to_action_set[0]
#%%

#%%
# inspect 8, 15
particle_belief = planner.mcts_planner.tree.belief_id_to_belief[2]
get_value_from_hj(particle_belief, hj_planner=hj_planner)
# get_value_from_hj_partial(particle_belief)
#%% values somehow go above 100, how is that possible?
planner.mcts_planner.reward_function.get_reward(particle_belief.states)
#%%
planner.mcts_planner._rollout_belief_simulation(0)
#%% do rollout manually!
# planner.mcts_planner._rollout_belief_simulation(1)
planner.mcts_planner._rollout_mdp_simulation(0)
# it doesn't reach the target from there, why
#%% Narrow it down by doing one step in both of them
rollout_belief_mdp = deepcopy(planner.mcts_planner.tree.belief_id_to_belief[0])
print(rollout_belief_mdp.states)
# now one step
actions = planner.mcts_planner.rollout_policy.get_action(rollout_belief_mdp.states)
print("actions", actions)
rollout_belief_mdp.update_states(
    planner.mcts_planner.generative_particle_filter.dynamics_and_observation.get_next_states(
        states=rollout_belief_mdp.states,
        actions=actions
    )
)
print(rollout_belief_mdp.states)
#%% now same for the other pathway
rollout_belief = deepcopy(planner.mcts_planner.tree.belief_id_to_belief[0])
print(rollout_belief.states)
# now one step
action = planner.mcts_planner._rollout_action(rollout_belief) # difference here it's sampled for ONE state, not for all particles!
print("actions", action)
rollout_belief, reward = planner.mcts_planner._generate_transition(rollout_belief, action)
print(rollout_belief.states)
#%%

#%%
traj = np.array(arena.state_trajectory)
#%%
import matplotlib.pyplot as plt
plt.plot(traj[:,0], traj[:,1], marker='x')
plt.show()
#%% check the actions
rollout_belief = planner.mcts_planner.tree.belief_id_to_belief[0]
# now along the state
for state in planner.mcts_planner.trajectory:
    rollout_belief.states = state
    action = planner.mcts_planner._rollout_action(rollout_belief)
    print(action)
