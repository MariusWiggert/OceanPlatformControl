from tqdm import tqdm

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.DoubleGyreProblemFactory import DoubleGyreProblemFactory

import time

from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController

start = time.time()

factory = DoubleGyreProblemFactory()

for j in range(10):
    problem = factory.next_problem()
    arena = ArenaFactory.create(scenario_name='double_gyre')
    controller = NaiveToTargetController(problem=problem)
    observation = arena.reset(problem.start_state)

    for i in tqdm(range(5000)):
        action = controller.get_action(observation)
        observation = arena.step(action)
        if problem.is_done(observation.platform_state):
            print('Problem solved!')
            break

    ax = arena.plot_spatial(
        ax=None if j==0 else ax,
        background='currents' if j==0 else None,
        problem=problem,
        show_control=False,
        show_trajectory=True,
        margin=2,
    )

ax.get_figure().show()

print("Total Script Time: ", time.time() - start)

