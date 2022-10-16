import numpy as np
import scipy
import casadi as ca

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.problem_factories.FileProblemFactory import FileProblemFactory

problem_factory = FileProblemFactory(csv_file=f'/seaweed-storage/generation/3_increased_timeout/problems.csv')
problem = problem_factory.next_problem()

hindcast_planner = HJReach2DPlanner.from_saved_planner_state(
    folder=f'/seaweed-storage/generation/2_improved_planner/seed_{problem.extra_info["seed"]}/batch_{problem.extra_info["batch"]}/',
    problem=problem,
    verbose=10,
)

self = hindcast_planner
width_deg = 0.2
width = 5

out_t = problem.start_state.date_time.timestamp()
out_y = np.linspace(problem.start_state.lat.deg - width_deg / 2, problem.start_state.lat.deg + width_deg / 2, width)
out_x = np.linspace(problem.start_state.lon.deg - width_deg / 2, problem.start_state.lon.deg + width_deg / 2, width)

# self.interpolator = scipy.interpolate.RegularGridInterpolator(
#     points=(self.reach_times, self.grid.states[:, 0, 0], self.grid.states[0, :, 1]),
#     values=(self.all_values - self.all_values.min()) * (self.current_data_t_T - self.current_data_t_0) / 3600,
#     method='linear',
# )
#
mx, my = np.meshgrid(out_x, out_y)

hindcast_planner.interpolator((np.repeat(out_t, my.size), mx.ravel(), my.ravel())).reshape((width, width))

# hindcast_planner.interpolate_value_function_in_hours_on_grid()

# self.casadi_interpolant = ca.interpolant('value_function', 'linear', [
#     self.current_data_t_0 + self.reach_times,
#     self.grid.states[:, 0, 0],
#     self.grid.states[0, :, 1],
# ], self.all_values.ravel(order='F'))
#
# (ca.vertcat(sym_time, sym_lat_degree, sym_lon_degree)