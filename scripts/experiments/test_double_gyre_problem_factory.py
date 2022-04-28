from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.DoubleGyreProblemFactory import DoubleGyreProblemFactory


factory = DoubleGyreProblemFactory()

problem = factory.next_problem()

print(problem.start_state, problem.end_region, problem.radius)


arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='double_gyre')

arena.plot_spatial(
    problem=problem,
    margin=2,
).get_figure().show()

