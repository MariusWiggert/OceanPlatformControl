#%% This is an example of deterministic planning using the POMDP MCTS planner.
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
init_state = [5., 2, 0]
target_state = [5., 8.0]
target_radius = 0.5

# Ground truth simulation setup
u_highway = 0.5
F_max = 0.4
dt_sim = 0.1
timeout_of_simulation = 20     # in seconds

# about the observations (Positions are observations)
dt_obs = 0.1
obs_noise = 1e-3    #0.005


# Instantiate Truth Simulator
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

#%% Calculate the optimal (known currents)
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
    "discrete_actions": True,
    "T_goal_in_seconds": 20,
    "calc_opt_traj_after_planning": False,
    "use_geographic_coordinate_system": False,
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.05,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "platform_dict": arena.platform.platform_dict,
}
hj_planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
# % Run reachability planner
observation = arena.reset(platform_state=x_0)
action = hj_planner.get_action(observation=observation)
#%% For sanity check, look at Value function
hj_planner.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    time_to_reach=True, plot_in_h=False,
)
# # hj_planner.plot_reachability_snapshot_over_currents(rel_time_in_seconds=0, time_to_reach=True)
#%% Run HJ discretized closed-loop
observation = arena.reset(platform_state=x_0)
problem_status = arena.problem_status(problem=problem)
step = 0
while problem_status == 0:
    # Get action
    action = hj_planner.get_action(observation=observation)
    observation = arena.step(action)
    # update problem status
    problem_status = arena.problem_status(problem=problem)

arena.plot_all_on_map(problem=problem)
#% compute the time Reward Function
state_traj = arena.state_trajectory[:, :3]
reward_func_time = Rewards.TimeRewardFunction(target_state, target_radius)
reward_time = reward_func_time.get_reward(state_traj)
print(f"Reward time: {reward_time.sum() + 100}")
#%% Run discretized Naive to Target
naive_discretized = RolloutPolicy.NaiveToTarget(target_state)
observation = arena.reset(platform_state=x_0)
problem_status = arena.problem_status(problem=problem)
step = 0
while problem_status == 0:
    # Get action
    discrete_action = naive_discretized.get_action(states=np.array(observation.platform_state).reshape(1, -1))
    action = PlatformAction.from_discrete_action(discrete_action[0])
    observation = arena.step(action)
    # update problem status
    problem_status = arena.problem_status(problem=problem)

arena.plot_all_on_map(problem=problem)
#% compute reward
state_traj = arena.state_trajectory[:, :3]
reward_func_time = Rewards.TimeRewardFunction(target_state, target_radius)
reward_time = reward_func_time.get_reward(state_traj)
print(f"Reward time: {reward_time.sum() + 100}")
#%% Instantiate POMDP MCTS Planner
from ocean_navigation_simulator.controllers.pomdp_planners.RolloutValue import get_value_from_hj
from ocean_navigation_simulator.controllers.pomdp_planners.utilities.analytical_currents import highway_current_analytical
from functools import partial

#% Get initial particles
_key = jax.random.PRNGKey(2)
u_highway_samples = np.array([0.5]).reshape(-1,1)

# Settings for the Outside Filter
particle_filter_dict = {
    'dt_observations': dt_obs,
    'resample': False,
    'no_position_uncertainty': True,
    'n_euler_per_dt_dynamics': 50, # How many euler steps per dt_dynamics to take
}

# Settings for the Planner
mcts_dict = {
    'num_planner_particles': 1, # because it's the one true sample
    'dt_mcts': 0.1,
    'n_euler_per_dt_dynamics': 1,
    'n_states': 2, #x, y
    'n_actions': 9, # 8 cardinal directions and a "no action"
    # For evaluating the value of a leave of the tree after a few expansions
    # we can either run a rollout policy for all particles and weight the rewards
    # or we have a value function to evaluate for each particle and then weight accordingly
    'rollout_policy': None, #RolloutPolicy.NaiveToTarget(target_state),
    'rollout_value': partial(get_value_from_hj, hj_planner=hj_planner),
    'no_position_uncertainty': False,
    'resample': False,
    'reward_function': Rewards.TimeRewardFunction(target_state, target_radius), # Rewards.RewardFunction(target_state, target_radius),
    'mcts_settings': {
        # 100 is very low, the higher the more flushed out the tree
        "num_mcts_simulate": 10,  # number of simulate calls to MCTS (either explores children or new node) from root node.
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

particle_vel_func = partial(highway_current_analytical,
                            y_range_highway=arenaConfig['ocean_dict']['hindcast']['source_settings']['y_range_highway'])
pomdp_planner = PomdpPlanner(
    x_target=target_state,
    particle_vel_func=particle_vel_func,
    stoch_params=u_highway_samples,
    F_max=F_max,
    t_init=init_state[2], init_state=init_state[:2],
    obs_noise=obs_noise,
    key=_key,
    particle_filter_dict=particle_filter_dict,
    mcts_dict=mcts_dict)

#%% Run the MCTS once just for checking/intuition
observation = arena.reset(platform_state=x_0)
import time
start = time.time()
action = pomdp_planner.get_action(observation=observation)
end = time.time()
print("action", PlatformAction.to_discrete_action(action))
print(end - start)
#% Inspect the MCTS Planning Tree
# for bigger trees with labels when hovering over the nodes/edges
visualize.plot_tree_plotly(pomdp_planner.mcts_planner.tree, node_size=5, q_val_decimals=1, reward_decimals=1)
#%% Run MCTS POMDP planner closed-loop with outside data assimilation
observation = arena.reset(platform_state=x_0)
problem_status = arena.problem_status(problem=problem)
step = 0
while problem_status == 0:
    # Get action
    action = pomdp_planner.get_action(observation=observation)
    print("step: ", step, "planner_state:", pomdp_planner.mcts_planner.tree.belief_id_to_belief[0].states, "action: ", action.to_discrete_action())
    observation = arena.step(action)
    print("next Observation: ", observation.platform_state)
    pomdp_planner.assimilate(x_obs=np.array(observation.platform_state)[:2], action=action.to_discrete_action())
    step += 1
    # update problem status
    problem_status = arena.problem_status(problem=problem)

arena.plot_all_on_map(problem=problem)

#% compute reward
state_traj = arena.state_trajectory[:, :3]
reward_func_time = Rewards.TimeRewardFunction(target_state, target_radius)
reward_time = reward_func_time.get_reward(state_traj)
print(f"Reward time: {reward_time.sum() + 100}")