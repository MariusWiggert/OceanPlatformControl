import yaml

from src.utils import hycom_utils
from src.planners.ipopt_planner import IpoptPlannerFixCur
import os
from src.problem import Problem
from datetime import datetime, timedelta, timezone

project_dir = os.path.abspath(os.path.join(os.getcwd()))
# Set stuff up
nc_file = '2021_06_1-05_hourly.nc4'
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

# long around the vortex
x_0 = [-96.9, 22.8]
x_T = [-96.9, 22.2]
t_0 = datetime(2021, 6, 1, 12, 10, 10, tzinfo=timezone.utc)
# planner fixed time horizon
T_planner_in_h = 224
# Getting parameters from yaml file
controller_yaml = open("config/controller.yaml")
parsed_yaml_file = yaml.load(controller_yaml, Loader=yaml.FullLoader)
gen_settings = parsed_yaml_file["planner"]["gen_settings"]
specific_settings= {'n_dec_var': 100, 'T_goal_in_h': 20, 'deg_around_x0_xT_box': 0.8, 'direction': 'forward','initial_set_radius': 0.1}
# Step 1: set up problem
prob = Problem(x_0, x_T, t_0, nc_file, forecast_folder="data/forecast",config_yaml='platform.yaml')
prob.viz()
# Step 2: initialize & run the planner
ipopt_planner = IpoptPlannerFixCur(problem=prob, gen_settings=gen_settings, specific_settings=specific_settings)
ipopt_planner.plan(x_T)
ipopt_planner.plot_opt_results()

#%% Simulator
# Step 3: init the simulator
sim = Simulator(ipopt_planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')
#%%
# Step 4: run the simulator
sim.run(T_in_h=T_planner_in_h)

#%% Step 5: plot it
sim.plot_trajectory(gif_name='ipopt_fixed_cur', plotting_type='2D')
# sim.plot_trajectory(name='ipopt_fixed_cur', plotting_type='battery')
# sim.plot_trajectory(name='ipopt_fixed_cur_gif', plotting_type='gif')