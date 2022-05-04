import datetime
import numpy as np
import time
import math

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.Observer import Observer
from ocean_navigation_simulator.env.Platform import Platform, PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint
from ocean_navigation_simulator.env.controllers.UnmotorizedController import UnmotorizedController
from ocean_navigation_simulator.env.Problem import Problem

from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.utils import units
from scripts.experiments.class_gp import OceanCurrentGP

_DELTA_TIME_NEW_PREDICTION = datetime.timedelta(hours=1)
_DURATION_SIMULATION = datetime.timedelta(days=3)
_NUMBER_STEPS = int(math.ceil(_DURATION_SIMULATION.total_seconds()/_DELTA_TIME_NEW_PREDICTION.total_seconds()))
_N_DATA_PTS = 100 #TODO: FINE TUNE THAT
_N_BURNIN_PTS = 30#100 # Number of minimum pts we gather from a platform to use as observations
_MAX_STEPS_PREDICTION = 50




#for each hour:
    # get the forecasts
    # Give it to the observer
    # The observer compute using the GP




#%%

arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='current_highway_GP')
#arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='double_gyre_GP')
#arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='gulf_of_mexico')

#%%
#controller = NaiveToTargetController(problem=Problem(
#    start_state=platform_state,
#    end_region=end_region
#))

controller = NaiveToTargetController(problem=Problem(
    start_state=platform_state,
    end_region=end_region
))
print("controller created")
#%%
observation = arena.reset(platform_state)
print("reset arena done")
#%%
gp = OceanCurrentGP(arena.ocean_field)
observer = Observer(gp, arena)
print("gp and observer created")
#%%
start = time.time()

for i in range(_N_BURNIN_PTS):
    action = controller.get_action(observation)
    arena_obs = arena.step(action)
    print("time after state:",arena_obs.platform_state.date_time)
    #true_current, forecast_current = arena_obs.true_current_at_state, arena_obs.forecasted_current_at_state
    #observer.observe(observation.platform_state, forecast_current.subtract(true_current))
    #print("Burnin step:{}/{}, position platform:{}, difference_observation:{}"
    #      .format(i + 1, _N_BURNIN_PTS, arena_obs.platform_state.to_spatial_point(),
    #              forecast_current.subtract(true_current)))
print("end of burnin time.")
arena.quick_plot(end_region)
#%% Predict at each step
print("start predicting")
for i in range(min(_NUMBER_STEPS,_MAX_STEPS_PREDICTION)):
    print("step {}/{}".format(i+1,_NUMBER_STEPS))
    action = controller.get_action(observation)
    arena_obs = arena.step(action)
    true_current, forecast_current = arena_obs.true_current_at_state, arena_obs.forecasted_current_at_state
    # Give as input to observer: position and forecast-groundTruth
    observer.observe(observation.platform_state, forecast_current.subtract(true_current))
    forecasts_mean, forecast_current_std = observer.fit_and_evaluate(platform_state)
    print("means:", forecasts_mean.shape, " std:", forecast_current_std.shape)
    #get_forecasts(observation.platform_state)
print("total: ", time.time() - start)
# Testing if solar caching or not-caching makes much of a difference
# For 240 steps: without caching 0.056s > with caching: 0.037.
#%%
#arena.do_nice_plot(x_T=np.array([controller.problem.end_region.lon.deg, controller.problem.end_region.lat.deg]))
arena.quick_plot(end_region)

#%%
print("over")
