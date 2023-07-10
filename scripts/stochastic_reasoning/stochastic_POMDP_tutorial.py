#%% This is an example of using the POMDP planner in a stochastic setting
# currently it's super manual to get the value function for the various particles/ensembles.
# that needs to be change to scale it =)
import jax
import datetime
import numpy as np

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
init_state = [5., 2.0, 0]
target_state = [5., 8.0]
target_radius = 0.5

# Ground truth simulation setup
u_highway_true = 0.5
F_max = 0.4
dt_sim = 0.1
timeout_of_simulation = 20     # in seconds

# the ensembles/particles
# Note: currently super manual!
u_highway_samples = np.linspace(-0.5, 0.5, 3).reshape(-1, 1)

# about the observations
dt_obs = 0.5
obs_noise = 1e-3#0.005


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
                    "spatial_resolution": 0.1,
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
)
#%% as we have 3 particles we'll have to get 3 value functions.
# we do it here super manually with the 3 HJ planners, needs to be done differently at scale.

# Particle 1: Get true optimal control
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
    "discrete_actions": True,
    "use_geographic_coordinate_system": False,
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.05,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "platform_dict": true_arena.platform.platform_dict,
}
hj_planner_true = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
# Run reachability planner to get the value function
observation = true_arena.reset(platform_state=x_0)
action = hj_planner_true.get_action(observation=observation)

# Particle 2: Get highway 0.0 optimal value function
u_highway = 0.0

zero_arenaConfig = {
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
zero_arena = ArenaFactory.create(scenario_config=zero_arenaConfig)
hj_planner_zero = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
# Run reachability planner to get the value function
observation = zero_arena.reset(platform_state=x_0)
action = hj_planner_zero.get_action(observation=observation)

# Particle 3: Get highway -0.5 optimal value function
u_highway = -0.5
minus_arenaConfig = {
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
minus_arena = ArenaFactory.create(scenario_config=minus_arenaConfig)
hj_planner_minus = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
# Run reachability planner to get the value function
observation = minus_arena.reset(platform_state=x_0)
action = hj_planner_minus.get_action(observation=observation)

#%% #% Now we can instantite the pomdp planner

# First we instantiate the function that can compute the value of a particle belief state
# essentially a location and weighting of the 3 particles.
from ocean_navigation_simulator.controllers.pomdp_planners.RolloutValue import get_value_from_hj_dict
from functools import partial

get_value_from_hj_partial = partial(get_value_from_hj_dict, hj_planner_dict={
    'true': hj_planner_true, 'zero': hj_planner_zero, 'minus': hj_planner_minus})

_key = jax.random.PRNGKey(2)
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
    'dt_mcts': 0.3,
    'n_euler_per_dt_dynamics': 10,
    'n_states': 2,
    'n_actions': 9,
    'rollout_policy': None, #RolloutPolicy.NaiveToTarget(target_state),
    'rollout_value': get_value_from_hj_partial,
    'no_position_uncertainty': True,
    'resample': False,
    'reward_function': Rewards.TimeRewardFunction(target_state, target_radius), # Rewards.RewardFunction(target_state, target_radius),
    'mcts_settings': {
        # 100 is very low, the higher the more flushed out the tree
        "num_mcts_simulate": 200,  # number of simulate calls to MCTS (either explores children or new node) from root node.
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

from ocean_navigation_simulator.controllers.pomdp_planners.utilities.analytical_currents import highway_current_analytical

particle_vel_func = partial(highway_current_analytical,
                            y_range_highway=arenaConfig['ocean_dict']['hindcast']['source_settings']['y_range_highway'])

# start in the middle, otherwise no signal in the observation
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
#%%
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
    print("step: ", step, "planner_state:", pomdop_planner.mcts_planner.tree.belief_id_to_belief[0].states, "action: ", action.to_discrete_action())
    for _ in range(int(dt_obs / dt_sim)):
        observation = true_arena.step(action)
    # observation = true_arena.step(action)
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