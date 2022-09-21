from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.utils import units

# Water Start
problem_factory = FileMissionProblemFactory()
problem = problem_factory.next_problem()
arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_hindcast', problem=problem)
arena.reset(problem.start_state)
print('Is on land:', arena.is_on_land())

# Somewhere inside Florida
problem.start_state.lon = units.Distance(deg=-81.421354)
problem.start_state.lat = units.Distance(deg=27.344535)
arena.reset(problem.start_state)
print('Is on land:', arena.is_on_land())