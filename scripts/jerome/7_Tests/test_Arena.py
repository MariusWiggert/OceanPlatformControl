from tqdm import tqdm
import time

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.problem_factories.DoubleGyreProblemFactory import DoubleGyreProblemFactory

start = time.time()

arena = ArenaFactory.create(scenario_name='double_gyre')
# arena = ArenaFactory.create(scenario_name='current_highway')
# arena = ArenaFactory.create(scenario_name='gulf_of_mexico')

#arena = ArenaFactory.create(scenario_name='double_gyre')

factory = DoubleGyreProblemFactory()
problem = factory.next_problem()
controller = NaiveController(problem=problem)
observation = arena.reset(problem.start_state)

for i in tqdm(range(5000)):#6 * 40)):
    action = controller.get_action(observation)
    observation = arena.step(action)

arena.plot_all_on_map(
    problem=problem,
    x_interval=[0,1.99],
    y_interval=[0,0.99]
).get_figure().show()

print(f"Total Script Time: {time.time() - start:.2f}s")