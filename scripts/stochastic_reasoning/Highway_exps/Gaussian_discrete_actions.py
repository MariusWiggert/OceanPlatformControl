#%% This is an example of using the POMDP planner in a stochastic setting
import jax
import datetime
import numpy as np
import jax.numpy as jnp
from functools import partial

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
y_highway = [4, 6]
F_max = 0.4
dt_sim = 0.1    # euler steps
timeout_of_simulation = 20     # in seconds

# the ensamble (used by the particle filter outside loop)
n_mc = 10
u_highway_samples = np.random.normal(loc=0, scale=0.5, size=(n_mc, 1))

# about the observations
dt_obs = 0.5
obs_noise = 1e-3 #0.005 # => very small will know true one after one observation in flow

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
                    "y_range_highway": y_highway,
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

#% Now we can instantite the pomdp planner
# Step 1: Set up the value function estimator for the POMDP planner
value_estimator_settings = {
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
highway_vel_fixed = partial(highway_vel, y_range_highway=y_highway)
value_estimator = ParticleHJValueEstimator(problem, value_estimator_settings, vel_func=jax.jit(highway_vel_fixed))
_key = jax.random.PRNGKey(10)

# Step 2: Set up the particle vel_func that is used inside for parallelized planning in the MCTS
from ocean_navigation_simulator.controllers.pomdp_planners.utilities.analytical_currents import highway_current_analytical
particle_vel_func = partial(highway_current_analytical,
                            y_range_highway=arenaConfig['ocean_dict']['hindcast']['source_settings']['y_range_highway'])

#%% Instantiate the planner
# Dicts for Particle Filter and MCTS
particle_filter_dict = {
    'dt_observations': dt_obs,
    'resample': False,
    'no_position_uncertainty': True,
    'n_euler_per_dt_dynamics': 50,
}
mcts_dict = {
    'num_planner_particles': n_mc, # manually set right now...
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

#%% Run POMDP planner closed-loop
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
# Right now it is just -1 per step of simulation (so 0.1 time units)
state_traj = true_arena.state_trajectory[:, :3]
reward_func_time = Rewards.TimeRewardFunction(target_state, target_radius)
reward_time = reward_func_time.get_reward(state_traj)
print(f"Reward time: {reward_time.sum() + 100}")
# Note: Optimal action is  straight (hedging) until first observation.
# then outside particle filter is converged and does the optimal action
# using the weighted HJ Value function.

#%% Notes below
# Why weights 1,1,1?
# Weird that the parameters were always the same...








#%% Some helper functions for debugging
print(pomdop_planner.particle_observer.particle_belief_state.states)
print(pomdop_planner.particle_observer.particle_belief_state.weights)
#%% inspect the belief state of the particle filter
visualize.plot_particles_in_1D(
    particle_belief=pomdop_planner.particle_observer.particle_belief_state,
     particle_axis_idx=3,
     particle_axis_label="u_highway",
     true_state=u_highway_true)
#%% Plot latest tree
visualize.plot_tree_plotly(pomdop_planner.mcts_planner.tree, node_size=5, q_val_decimals=1, reward_decimals=1)
#%% Let's look at belief in the tree
belief_state_id = 15
visualize.plot_belief_in_tree_1D(pomdop_planner.mcts_planner.tree, belief_state_id, true_state=u_highway_true)
#%% visualize both TTR functions
value_estimator.plot_ttr_snapshot(stoch_params_idx=0, time_idx=-1, granularity=2, time_to_reach=True)


