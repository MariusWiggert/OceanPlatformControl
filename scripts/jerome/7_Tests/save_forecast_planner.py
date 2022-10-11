from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory

problem_factory = FileMissionProblemFactory()
problem = problem_factory.next_problem()

arena = ArenaFactory.create(scenario_name='gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast', problem=problem)

controller = NaiveController(problem=problem)

observation = arena.reset(problem.start_state)

while arena.problem_status(problem=problem) == 0:
	observation = arena.step(controller.get_action(observation))
	print(observation)