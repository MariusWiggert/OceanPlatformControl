from c3python import C3Python
from ocean_navigation_simulator.problem_factories.MissionProblemFactory import MissionProblemFactory
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.reinforcement_learning.OceanFeatureConstructor import OceanFeatureConstructor


problem_factory = MissionProblemFactory()
problem = problem_factory.next_problem()
feature_constructor = OceanFeatureConstructor()
arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_hindcast', problem=problem)

observation = arena.reset(problem.start_state)
feature_constructor.get_features_from_state(obs=observation, problem=problem)