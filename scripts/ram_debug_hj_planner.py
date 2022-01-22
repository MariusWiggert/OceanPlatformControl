import sys
sys.path.extend(['/Volumes/Data/2_Work/2_Graduate_Research/1_Seaweed/OceanPlatformControl'])
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import OceanNavSimulator
from ocean_navigation_simulator.planners import HJReach2DPlanner
import numpy as np
import datetime
import os
import hj_reachability as hj
import time
from jax.interpreters import xla


@profile
def main():
    platform_config_dict = {'battery_cap': 500.0, 'u_max': 0.2, 'motor_efficiency': 1.0,
                            'solar_panel_size': 0.5, 'solar_efficiency': 0.2, 'drag_factor': 675}

    t_0 = datetime.datetime(2021, 11, 22, 10, 00, 10, tzinfo=datetime.timezone.utc)
    x_0 = [-94.5, 25., 1]  # lon, lat, battery
    x_T = [-83.0, 24.8] # upward eddy before florida
    hindcast_folder = "data/hindcast_test/"
    forecast_folder = "data/forecast_test/"
    forecast_delay_in_h = 0.
    plan_on_gt=False
    prob = Problem(x_0, x_T, t_0,
                   platform_config_dict=platform_config_dict,
                   hindcast_folder= hindcast_folder,
                   forecast_folder=forecast_folder,
                   plan_on_gt = plan_on_gt,
                   x_T_radius = .5,
                   forecast_delay_in_h=forecast_delay_in_h)


    sim = OceanNavSimulator(sim_config_dict="simulator.yaml", control_config_dict='reach_controller.yaml', problem=prob)
    # check to update current data in the dynamics
    sim.check_dynamics_update()

    # Round 1:
    current_forecast_dict_idx = sim.problem.most_recent_forecast_idx
    sim.high_level_planner.update_forecast_dicts([sim.problem.forecasts_dicts[current_forecast_dict_idx]])
    if sim.high_level_planner.new_forecast_dicts:
        sim.high_level_planner.update_current_data(x_t=sim.cur_state)
    sim.high_level_planner.plan(sim.cur_state, trajectory=None)
    print("solve cache after round 1: ", hj.solver._solve._cache_size())
    # How can I delete the previous cache?
    hj.solver._solve._clear_cache()
    # del hj.solver._solve
    xla._xla_callable.cache_clear()

    # go until replan is needed
    while sim.cur_state[3] < sim.problem.forecasts_dicts[current_forecast_dict_idx + 1]['t_range'][0].timestamp():
        print("run step")
        sim.run_step()

    # Round 2:
    current_forecast_dict_idx += 1
    sim.high_level_planner.update_forecast_dicts([sim.problem.forecasts_dicts[current_forecast_dict_idx]])
    if sim.high_level_planner.new_forecast_dicts:
        sim.high_level_planner.update_current_data(x_t=sim.cur_state)
    sim.high_level_planner.plan(sim.cur_state, trajectory=None)

    print("solve cache after round 2: ", hj.solver._solve._cache_size())

    hj.solver._solve._clear_cache()
    # del hj.solver._solve
    xla._xla_callable.cache_clear()

    # go until replan is needed
    while sim.cur_state[3] < sim.problem.forecasts_dicts[current_forecast_dict_idx + 1]['t_range'][0].timestamp():
        print("run step")
        sim.run_step()

    # Round 3:
    current_forecast_dict_idx += 1
    sim.high_level_planner.update_forecast_dicts([sim.problem.forecasts_dicts[current_forecast_dict_idx]])
    if sim.high_level_planner.new_forecast_dicts:
        sim.high_level_planner.update_current_data(x_t=sim.cur_state)
    sim.high_level_planner.plan(sim.cur_state, trajectory=None)

    print("solve cache after round 3: ", hj.solver._solve._cache_size())

    hj.solver._solve._clear_cache()
    # del hj.solver._solve
    xla._xla_callable.cache_clear()


if __name__ == '__main__':
    main()


