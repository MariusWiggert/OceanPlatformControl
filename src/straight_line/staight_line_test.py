from src.straight_line.straight_line_planner import StraightLinePlanner
from src.utils.classes import *
from src.utils import hycom_utils
from src.non_lin_opt.ipopt_planner import IpoptPlanner
import os
project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)
u_max = 1.

# Set starting positions
x_0 = [-96.6, 22.8]
x_T = [-97.5, 22.2]

# planner fixed time horizon
T_planner = 116500

# Step 1: set up problem
prob = Problem(fieldset, x_0, x_T, u_max)
# prob.viz()
#%%
# Step 2: initialize planner
planner = StraightLinePlanner(problem=prob, t_init=T_planner)
#%%
# Step 3: init the simulator
settings = {'dt': planner.dt, 'conv_m_to_deg': 111120., 'int_pol_type': 'bspline', 'sim_integration': 'rk',
            'project_dir': project_dir}
sim = Simulator(planner, problem=prob, settings=settings)
#%%
# Step 4: run the simulator
sim.run(T_planner)
#%%
# Step 5: plot it
sim.plot_trajectory(name='classes_test', plotting_type='2D')