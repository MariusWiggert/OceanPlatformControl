import sys
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import OceanNavSimulator
from ocean_navigation_simulator import utils
from ocean_navigation_simulator.planners import HJReach2DPlanner
import numpy as np
import datetime
import os
import hj_reachability as hj
import time
from scipy.interpolate import interp1d


# new imports added
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial


#% Settings to feed into the planner
# Set the platform configurations
platform_config_dict = {'battery_cap': 400.0, 'u_max': 0.1, 'motor_efficiency': 1.0,
                        'solar_panel_size': 0.5, 'solar_efficiency': 0.2, 'drag_factor': 675}

# Create the navigation problem
t_0 = datetime.datetime(2021, 11, 22, 12, 10, 10, tzinfo=datetime.timezone.utc)
x_0 = [-88.25, 26.5, 1]  # lon, lat, battery
# to check for robust reachability
x_0 = [-88.75, 27.25, 1]  # lon, lat, battery
x_T = [-90, 26]
hindcast_folder = "data/single_day_hindcasts/"
# hindcast_folder = "data/hindcast_to_extend/"
forecast_folder = "data/forecast_test/"
forecast_delay_in_h = 0.
plan_on_gt=False
prob = Problem(x_0, x_T, t_0,
               platform_config_dict=platform_config_dict,
               hindcast_source= hindcast_folder,
               forecast_source=forecast_folder,
               plan_on_gt = plan_on_gt,
               x_T_radius = 0.1,
               forecast_delay_in_h=forecast_delay_in_h)
#%%
prob.viz() # plots the current at t_0 with the start and goal position
# # create a video of the underlying currents rendered in Jupyter, Safari or as file
# prob.viz(video=True) # renders in Jupyter
# prob.viz(video=True, html_render='safari')    # renders in Safari
# prob.viz(video=True, filename='problem_animation.gif', temp_horizon_viz_in_h=200) # saves as gif file
#%%
specific_settings = {
      'direction': 'forward-backward',
      'd_max': 0.05,
      'T_goal_in_h': 90,
      'n_time_vector': 100,
      'initial_set_radii': [0.1, 0.1],
      'fwd_back_buffer_in_h': 3,
      'grid_res': (0.04, 0.04),
      'deg_around_xt_xT_box': 2.,
      'accuracy': 'high',
      'artificial_dissipation_scheme': 'local_local',
      'progress_bar': True}
planner = HJReach2DPlanner(prob, specific_settings, conv_m_to_deg=111120.)
planner.forecast_data_source = [prob.forecast_data_source[0]]

x_t = np.array(x_0 + [t_0.timestamp()])
planner.plan(x_t)
#% if forward
# _, t_earliest_in_h = planner.get_t_earliest_for_target_region()
# print(t_earliest_in_h) # was 79.2h with 0.05 m/s disturbance. With 0 disturbance it's 76.5h
#%%
planner.plot_2D_traj()
planner.plot_ctrl_seq()
#%% Visualize a snapshot of it
rel_time_in_h = 50 # should works from 0.1 to 99.9
planner.plot_reachability_snapshot(rel_time_in_h,  multi_reachability=False, granularity_in_h=10, time_to_reach=True)
#%% animate reachability
planner.plot_reachability_animation(type='mp4', multi_reachability=False, granularity_in_h=5, time_to_reach=False)
#%% Test closed-loop performance
sim = OceanNavSimulator(sim_config_dict="simulator.yaml", control_config_dict='robustness_ctrl.yaml', problem=prob)
start = time.time()
sim.run(T_in_h=100)
print("Took : ", time.time() - start)

#%% Step 5: plot from Simulator
# # plot Battery levels over time
sim.plot_trajectory(plotting_type='battery')
# # # plot 2D Trajectory without background currents
sim.plot_trajectory(plotting_type='2D')
# # plot control over time
sim.plot_trajectory(plotting_type='ctrl')
# # plot 2D Trajectory with currents at t_0
sim.plot_trajectory(plotting_type='2D_w_currents')
# # plot 2D Trajectory with currents at t_0 and control_vec for each point
sim.plot_trajectory(plotting_type='2D_w_currents_w_controls')
#%% plot simulator animation
# # render in Jupyter
# # sim.plot_trajectory(plotting_type='video')
# render in Safari
# sim.plot_trajectory(plotting_type='video', html_render='safari')
# save as gif file
sim.plot_trajectory(plotting_type='video', vid_file_name='sim_animation.gif')
#%% Plot from reachability planner
sim.high_level_planner.plot_2D_traj()
sim.high_level_planner.plot_ctrl_seq()
#%% Plot 2D reachable set evolution
sim.high_level_planner.plot_reachability(type='gif')




#%% Setting for both & Initialize planner
planner = HJReach2DPlanner(prob, specific_settings, conv_m_to_deg=111120.)
planner.forecast_data_source = [prob.forecast_data_source[0]]
#%% Forward-Backward approach
specific_settings['direction'] = 'forward-backward'
planner.plan(x_t)
#%% plot stuff
# planner.plot_2D_traj()
# planner.plot_ctrl_seq()

#%% Now with multi-reachability instead of forward-backward imperfection
specific_settings['direction'] = 'multi-reach'
# Do everything that would happen in plan:
planner.update_current_data(x_t=x_t)
x_t_rel = np.copy(x_t)
x_t_rel[3] = x_t_rel[3] - planner.current_data_t_0


# Set-Up before in the run_hj_reachability function
initial_values=planner.get_initial_values(center=planner.x_T, direction='multi-reach')
t_start=x_t_rel[3]
T_max_in_h=planner.specific_settings['T_goal_in_h']
dir='multi-reach'
x_reach_end=None
stop_at_x_end=False

#%
# Step 1: Run Multi-reachability from fixed time-Horizon backwards
# TODO: Maybe do it more robustly by not planning a fixed time in advance but min(fixed_time, max_time_of_file)

# set the time_scales and offset in the non_dim_dynamics in which the PDE is solved
# If we want to do this, the parameter becomes 'T_max_look_ahead'.
max_forecast_horizon = planner.current_data_t_T - planner.current_data_t_0
#%
planner.nondim_dynamics.tau_c = min(T_max_in_h * 3600, int(planner.current_data_t_T - planner.current_data_t_0))
planner.nondim_dynamics.t_0 = t_start

# set up the non_dimensional time-vector for which to save the value function
solve_times = np.linspace(0, 1, planner.specific_settings['n_time_vector'] + 1)
# solve_times = t_start + np.linspace(0, T_max_in_h * 3600, self.specific_settings['n_time_vector'] + 1)

if dir == 'backward' or dir == 'multi-reach':
    solve_times = np.flip(solve_times, axis=0)
    planner.nondim_dynamics.dimensional_dynamics.control_mode = 'min'
elif dir == 'forward':
    planner.nondim_dynamics.dimensional_dynamics.control_mode = 'max'

if dir == 'multi-reach':
    # write multi_reach hamiltonian postprocessor
    def multi_reach_step(mask, val):
        val = jnp.where(mask <= 0, -1, val)
        return val
    # combine it with partial sp the mask input gets fixed and only val is open
    p_multi_reach_step = partial(multi_reach_step, initial_values)
    # set the postprocessor to be fed into solver_settings
    hamiltonian_postprocessor = p_multi_reach_step
    print("running multi-reach")
else: # make the postprocessor the identity
    hamiltonian_postprocessor = lambda *x: x[-1]
#% viz the mask of the postprocessor
# plt.imshow((initial_values <=0).T)
# plt.show()
# Note: Seems like we need a good enough resolution so that the inside mask is not too small!
#%
# set variables to stop or not when x_end is in the reachable set
if stop_at_x_end:
    stop_at_x_init = planner.get_non_dim_state(x_reach_end)
else:
    stop_at_x_init = None

# create solver settings object
solver_settings = hj.SolverSettings.with_accuracy(accuracy=planner.specific_settings['accuracy'],
                                                  # x_init=stop_at_x_init,
                                                  artificial_dissipation_scheme=planner.diss_scheme,
                                                  hamiltonian_postprocessor=hamiltonian_postprocessor)

# solve the PDE in non_dimensional to get the value function V(s,t)
non_dim_reach_times, planner.all_values = hj.solve(
    solver_settings=solver_settings,
    dynamics=planner.nondim_dynamics,
    grid=planner.nonDimGrid,
    times=solve_times,
    initial_values=initial_values,
    progress_bar=planner.specific_settings['progress_bar']
)

# scale up the reach_times to be dimensional_times in seconds again
planner.reach_times = non_dim_reach_times * planner.nondim_dynamics.tau_c + planner.nondim_dynamics.t_0
#%% Visualize a snapshot of it
rel_time_in_h = -100
planner.plot_reachability_snapshot(rel_time_in_h,  multi_reachability=True, granularity_in_h=5, time_to_reach=True)
#%% animate reachability
planner.plot_reachability_animation(type='mp4', multi_reachability=True, granularity_in_h=5, time_to_reach=False)
#%% check value at
if planner.grid.interpolate(planner.all_values[0, ...], [-88.25,  26.5 ]) > 0:
    print("x_init is not in the Backwards/Forwards Reachable Set/Tube")
#%% Extract trajectory
# Essentially just need to release the vehicle at the current time (meaning latest time in reachability) and run forwards.
# planner.flip_value_func_to_forward_times()
# Now just extract it forwards releasing the vehicle at t=0
def termination_condn(x_target, r, x, t):
    return jnp.linalg.norm(x_target - x) <= r
termination_condn = partial(termination_condn, planner.x_T, planner.problem.x_T_radius)
planner.extract_trajectory(planner.get_x_from_full_state(x_t_rel.flatten()),
                           traj_rel_times_vector=None, termination_condn=termination_condn)

#%%
hj.viz.visTrajSet2D(planner.x_traj, planner.grid, planner.all_values[:74,...],
                    planner.times, colorbar=False, t_unit='h')
#%%
# arrange to forward times by convention for plotting and open-loop control
planner.flip_traj_to_forward_times()
planner.flip_value_func_to_forward_times()
#%% Test if get_next action points in the right direction after the flipping around
state = np.array(prob.x_0).reshape(4,-1)
u_out = planner.get_next_action(state)

#%%
print("u_x = ", np.cos(u_out[1]))
print("u_y = ", np.sin(u_out[1]))
#%%
# planner.plot_2D_traj()
# planner.flip_traj_to_forward_times()
# planner.plot_ctrl_seq()

#%%
hj.viz.visTrajSet2D(planner.x_traj, planner.grid, np.flip(planner.all_values, axis=0),
                    planner.times, colorbar=False, t_unit='h')
#%%

# we 'release' the vehicle at t=-6s and let it follow the optimal control until t=0
traj_times, x_traj_fine, contr_seq, distr_seq = Plat2D.backtrack_trajectory(grid, x_init, times, all_values,
                                                                            traj_times=np.linspace(0, -6, 61),
                                                                            termination_condn=termination_condn)
# %% Plot the Traj on the set
hj.viz.visTrajSet2D(x_traj_fine, grid, all_values, times, x_init=x_init, colorbar=False, t_unit='h')
# %% Plot the Control Traj
plt.plot(traj_times[1:], contr_seq[0, :], color='r')
plt.plot(traj_times[1:], contr_seq[1, :], color='b')
plt.show()


# the value function along the trajectory here should be stable (except for numerical error it is)
# The value means: I start at -6 and because value is -4, I will arrive at time t=-4.25 when going forward in time.
val_at_t = interp1d(times[::-1], all_values[::-1], axis=0, kind='linear', assume_sorted=True)
val_at_traj_fine = np.zeros_like(x_traj_fine)
vals_at_traj_fine = np.array([grid.interpolate(val_at_t(time), x) for x, time in zip(x_traj_fine.T, traj_times)])

fig,ax = plt.subplots()
ax.plot(traj_times, vals_at_traj_fine,  ls='-', marker='.', label="Fine")
ax.set_ylabel(r"$\phi(x_t)$", fontsize=16)
ax.set_xlabel(r"$t$", fontsize=16)
ax.set_title("Value function along the vehicle trajectory", fontsize=20)
ax.grid()
ax.set_ylim([-2,-8])
ax.axhline(traj_times[0],color='k', ls='--')
plt.show()

#%%
# elif self.specific_settings['direction'] == 'backward':
# # Note: no trajectory is extracted as the value function is used for closed-loop control
# self.run_hj_reachability(initial_values=self.get_initial_values(center=self.x_T, direction="backward"),
#                          t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
#                          dir='backward')
# self.extract_trajectory(x_start=self.get_x_from_full_state(x_t_rel.flatten()), traj_rel_times_vector=None)
# # arrange to forward times by convention for plotting and open-loop control
# self.flip_traj_to_forward_times()
# self.flip_value_func_to_forward_times()
# elif self.specific_settings['direction'] == 'forward-backward':
# # Step 1: run the set forward to get the earliest possible arrival time
# self.run_hj_reachability(initial_values=self.get_initial_values(center=x_t_rel, direction="forward"),
#                          t_start=x_t_rel[3], T_max_in_h=T_max_in_h,
#                          dir='forward', x_reach_end=self.get_x_from_full_state(self.x_T), stop_at_x_end=True)
# # Step 2: run the set backwards from the earliest arrival time backwards
# _, t_earliest_in_h = self.get_t_earliest_for_target_region()
# print("earliest for target region is ", t_earliest_in_h)
# self.run_hj_reachability(initial_values=self.get_initial_values(center=self.x_T, direction="backward"),
#                          t_start=x_t_rel[3],
#                          T_max_in_h=t_earliest_in_h + self.specific_settings['fwd_back_buffer_in_h'],
#                          dir='backward')
# self.extract_trajectory(x_start=self.get_x_from_full_state(x_t_rel.flatten()), traj_rel_times_vector=None)
# # arrange to forward times by convention for plotting and open-loop control
# self.flip_traj_to_forward_times()
# self.flip_value_func_to_forward_times()
# else:
# raise ValueError("Direction in controller YAML needs to be one of {backward, forward, forward-backward}")
#
# # log it for plotting planned trajectory
# self.x_t = x_t
#%%
# # Step 1: run the set forward to get the earliest possible arrival time
# self.run_hj_reachability(initial_values=self.get_initial_values(center=x_t_rel, direction="forward"),
#                          t_start=x_t_rel[3], T_max_in_h=self.specific_settings['T_goal_in_h'],
#                          dir='forward', x_reach_end=self.get_x_from_full_state(self.x_T), stop_at_x_end=True)
# # Step 2: run the set backwards from the earliest arrival time backwards
# _, t_earliest_in_h = self.get_t_earliest_for_target_region()
# print("earliest for target region is ", t_earliest_in_h)
# self.run_hj_reachability(initial_values=self.get_initial_values(center=self.x_T, direction="backward"),
#                          t_start=x_t_rel[3],
#                          T_max_in_h=t_earliest_in_h + self.specific_settings['fwd_back_buffer_in_h'],
#                          dir='backward')
# self.extract_trajectory(x_start=self.get_x_from_full_state(x_t_rel.flatten()), traj_rel_times_vector=None)
# # arrange to forward times by convention for plotting and open-loop control
# self.flip_traj_to_forward_times()
# self.flip_value_func_to_forward_times()











#%% Check feasibility of the problem
feasible, earliest_T, feasibility_planner = utils.check_feasibility2D(
    problem=prob, T_hours_forward=80, deg_around_xt_xT_box=2, progress_bar=True)
#%%
feasibility_planner.plot_2D_traj()
feasibility_planner.plot_ctrl_seq()
feasibility_planner.plot_reachability(type='gif')
#%% Set stuff up
sim = OceanNavSimulator(sim_config_dict="simulator.yaml", control_config_dict='reach_controller.yaml', problem=prob)
start = time.time()
sim.run(T_in_h=100)
print("Took : ", time.time() - start)
#%% Save as hdf5 file
import h5py
# Create a large dict with all trajectory data
trajs_dict = {"sim_traj":sim.trajectory,
              "sim_ctrl":sim.control_traj}

if len(sim.high_level_planner.planned_trajs) > 0:
    trajs_dict['plans'] = sim.high_level_planner.planned_trajs

# save as hdf5
# hf = h5py.File('data.h5', 'w')
# hf.create_dataset('dataset_1', data=trajs_dict, compression="gzip", compression_opts=5)
# hf.close()
import pickle
file_pi = open('filename_pi.obj', 'wb')
pickle.dump(trajs_dict, file_pi)

#%% open it
filehandler = open('filename_pi.obj', 'rb')
trajs_dict_loaded = pickle.load(filehandler)
#%%  plot all trajs in one plot
from ocean_navigation_simulator.utils import plot_2D_traj, plot_opt_ctrl
for plan in sim.high_level_planner.planned_trajs:
    # plot 2D
    plot_2D_traj(plan['traj'])
    #plot ctrl
    plot_opt_ctrl(plan['traj'][3,:-1], plan['ctrl'])
#%% Step 5: plot from Simulator
# # plot Battery levels over time
sim.plot_trajectory(plotting_type='battery')
# # # plot 2D Trajectory without background currents
sim.plot_trajectory(plotting_type='2D')
# # plot control over time
sim.plot_trajectory(plotting_type='ctrl')
# # plot 2D Trajectory with currents at t_0
sim.plot_trajectory(plotting_type='2D_w_currents')
# # plot 2D Trajectory with currents at t_0 and control_vec for each point
sim.plot_trajectory(plotting_type='2D_w_currents_w_controls')
#%% plot simulator animation
# # render in Jupyter
# # sim.plot_trajectory(plotting_type='video')
# render in Safari
# sim.plot_trajectory(plotting_type='video', html_render='safari')
# save as gif file
sim.plot_trajectory(plotting_type='video', vid_file_name='sim_animation.gif')
#%% Plot from reachability planner
sim.high_level_planner.plot_2D_traj()
sim.high_level_planner.plot_ctrl_seq()
#%% Plot 2D reachable set evolution
sim.high_level_planner.plot_reachability(type='gif')