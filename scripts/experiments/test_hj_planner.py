import datetime
import numpy as np
from tqdm import tqdm

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.utils import units
import matplotlib.pyplot as plt
import time
arena = ArenaFactory.create(scenario_name='gulf_of_mexico_files')

# #%% Plot to check if loading worked
# t_0 = datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc)
# t_interval = [t_0, t_0 + datetime.timedelta(days=4)]
# x_interval = [-82, -80]
# y_interval = [24, 26]
# # x_0 = PlatformState(lon=units.Distance(deg=-81.5), lat=units.Distance(deg=23.5), date_time=t_0)
# # x_T = SpatialPoint(lon=units.Distance(deg=-80), lat=units.Distance(deg=24.2))
# # Plot Hindcast
# arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(time=t_0 + datetime.timedelta(days=2),
#                                                                    x_interval=x_interval, y_interval=y_interval)
# # Plot Forecast at same time
# xarray_out = arena.ocean_field.forecast_data_source.get_data_over_area(t_interval=t_interval, x_interval=x_interval, y_interval=y_interval)
# ax = arena.ocean_field.forecast_data_source.plot_data_from_xarray(time_idx=49, xarray=xarray_out)
# plt.show()
forecast_data_source = arena.ocean_field.forecast_data_source
#%% Specify Problem
x_0 = PlatformState(lon=units.Distance(deg=-82), lat=units.Distance(deg=25),
                    date_time=datetime.datetime(2021, 11, 22, 12, 0, tzinfo=datetime.timezone.utc))
x_T = SpatialPoint(lon=units.Distance(deg=-80), lat=units.Distance(deg=24))
problem = Problem(start_state=x_0, end_region=x_T, target_radius=0.1)
#%% Plot the problem function -> To create

#%% Instantiate the HJ Planner

#%% Run the HJ Planner closed loop


# TODO: Think about how it works with analytical currents?