import yaml

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory
from ocean_navigation_simulator.reinforcement_learning.scripts import cluster_utils

cluster_utils.ensure_storage_connection()
problem_factory = FileMissionProblemFactory()
problem = problem_factory.next_problem()

arena = ArenaFactory.create(scenario_name='gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast', problem=problem, verbose=10)
arena_observation = arena.reset(problem.start_state)
controller = NaiveController(problem=problem)

with open(f'config/ocean_observer/config_GP_for_reinforcement_learning.yaml') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
observer = Observer(config['observer'])

WIDTH = 1
WIDTH_DEG = 0.05

for i in range(1):
	arena_observation = arena.step(controller.get_action(arena_observation))
	observer.observe(arena_observation)
	observer.fit()
	data = observer.get_data_over_area(
		x_interval=[arena_observation.platform_state.lon.deg-WIDTH_DEG/2, arena_observation.platform_state.lon.deg+WIDTH_DEG/2],
		y_interval=[arena_observation.platform_state.lat.deg-WIDTH_DEG/2, arena_observation.platform_state.lat.deg+WIDTH_DEG/2],
		t_interval=[arena_observation.platform_state.date_time, arena_observation.platform_state.date_time],
		spatial_resolution=WIDTH_DEG/WIDTH,
	)
	print(arena_observation)
	print(data)