#%% This is an example of using the POMDP planner in a stochastic setting
# add lines for auto-reloading external modules
import math
from typing import Tuple, Callable, Dict

from matplotlib import pyplot as plt

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
%load_ext autoreload
%autoreload 2

import jax
import datetime
import numpy as np
import jax.numpy as jnp
from ocean_navigation_simulator.controllers.pomdp_planners.HJRolloutValue.ParticleHJValueEstimator import ParticleHJValueEstimator

from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units


# initialize problem
# Start - Goal Settings
# for highway case
init_state = [5., 2.0, 0]
target_state = [5., 8.0]
target_radius = 1.0
x_range=[0, 10]
y_range=[0, 10]
# # for double gyre
# init_state = [0.1, 0.1, 0]
# target_state = [1.5, 0.8]
# target_radius = 0.1
# x_range=[0, 2]
# y_range=[0, 1]

# Ground truth simulation setup
F_max = 0.5
platform_dict = {
        "battery_cap_in_wh": 400.0,
        "drag_factor": 675.0,
        "dt_in_s": 0.1,
        "motor_efficiency": 1.0,
        "solar_efficiency": 0.2,
        "solar_panel_size": 0.5,
        "u_max_in_mps": F_max}

# Specify Navigation Problem
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
    platform_dict=platform_dict,
    x_range=x_range,
    y_range=y_range
)

specific_settings = {
        "n_time_vector": 200,
        "T_goal_in_units": 10,
        "discrete_actions": False,
        "grid_res": 0.05,
        "ttr_to_rewards" : lambda x: -(x * 10)  # + 100
}
#%%
def anlt_dg_vel(t, x, y, A, eps, omega):
    a = eps * jnp.sin(omega * t)
    b = 1 - 2 * a
    f = a * (x ** 2) + b * x
    df = 2 * (a * x) + b
    return jnp.array([-jnp.pi * A * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * y),
                      jnp.pi * A * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * y) * df])

def highway_vel(t, x, y, u_highway, y_range_highway):
    u_cur_out = jnp.where(jnp.logical_and(y_range_highway[0] <= y, y <= y_range_highway[1]), u_highway, 0.0)
    # adding 0 in the y direction
    return jnp.array([u_cur_out, 0.0])

# use functools partial to fix y_range_highway to [2,5]
from functools import partial
highway_vel_fixed = partial(highway_vel, y_range_highway=[2,5])

value_estimator = ParticleHJValueEstimator(problem, specific_settings, vel_func=jax.jit(highway_vel_fixed))
#%% visualize the initial value function, just to check
import hj_reachability as hj
hj.viz.visSet(grid=value_estimator.grid, data=value_estimator.term_val_func)
#%% parameters (double gyre)
_key = jax.random.PRNGKey(1)
key = _key
n_mc = 4

init_state_sd = 0.005
init_state_err = 0.5

F_max = 1.0
A,eps,ω = 0.4,0.3,2*np.pi/1.5
A_sd, eps_sd, ω_sd = 0.2, 0.2, 1.0
A_err, eps_err, ω_err = -0.9, 0.5, -0.5 #as multiples of sd

true_params = jnp.array([A, eps, ω])

A_st = (jax.random.normal(key,shape=[n_mc,1])*A_sd + A + A_err*A_sd)
key, _key = jax.random.split(_key)
eps_st = (jax.random.normal(key,shape=[n_mc,1])*eps_sd + eps + eps_err*eps_sd)
key, _key = jax.random.split(_key)
omega_st = (jax.random.normal(key,shape=[n_mc,1])*ω_sd + ω + ω_err*ω_sd)

params = jnp.concatenate([A_st,eps_st,omega_st],axis=1)
# shape is (4,3) so (n_mc, n_params)
#%% params for highway case
params = jnp.array([[0, 0.5, -0.5]]).reshape(3,1)
#% kick of the compute
current_time = init_state[2]
value_estimator.compute_hj_value_funcs(current_time, stoch_param=params)
#%% examine the value functions
value_estimator.plot_ttr_snapshot(stoch_params_idx=2, time_idx=-1, time_to_reach=True, granularity=5)
#%%
time = 0
state = np.array(init_state[:2])
value_estimator.grid.spacetime_interpolate(value_estimator.times, value_estimator.all_val_funcs[0,:,:,:], time, state)

#%% now estimate rollout value
# first create particle belief
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import ParticleBelief
# states are init_state and then stochastic parameters
# init_state is (3,) and params is (n_mc, n_params)
n_mc = params.shape[0]
particle_states = np.concatenate([np.tile(init_state, (n_mc, 1)), params], axis=1)
weights = np.ones(n_mc) / n_mc
particleBelief = ParticleBelief(states=particle_states, weights=weights)
#%
value_estimator.estimate_rollout_value(particleBelief)
#%% now estimate the rollout_value
# value_estimator.estimate_rollout_value(belief)
self = value_estimator
self.grid.spacetime_interpolate(self.times, self.all_val_funcs[1,...],
                                                     particleBelief.states[0, 2],  # times
                                                     particleBelief.states[0, :2]  # positions (x,y)
                                                     )
#%% using vmap from jax to compute the value for each particle
hj_values = jax.vmap(self.grid.spacetime_interpolate, in_axes=(None, 0, 0, 0), out_axes=0)(
    self.times, self.all_val_funcs, particleBelief.states[:, 2], particleBelief.states[:, :2])
#%%
# DO I need to transform them to time-to-reach? I guess not, didn't do it before...
# now upscale to same time-resolution as the reward by 10 and add 100 as final end reward.
# reward_values = -(hj_values * 10)  # + 100
reward_values = self.specific_settings['ttr_to_rewards'](hj_values)
# calculate the sum (because weights add to 1 already)
value_estimates = (reward_values * particleBelief.weights).sum()