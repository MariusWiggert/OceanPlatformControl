from tqdm import tqdm
import seaborn as sns
import os

os.environ['LOGLEVEL'] = 'ERROR'

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.controllers.RLController import RLController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.problem_factories.FileProblemFactory import FileProblemFactory
from ocean_navigation_simulator.utils.misc import timing

sns.set_theme()


with timing("Total Script Time: {:.2f}s", verbose=1):
    factory = FileProblemFactory(
        csv_file='/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/verification_1000_problems/problems.csv',
        # indices=[],
    )
    problem = factory.next_problem()

    arena = ArenaFactory.create(scenario_name='gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast', problem=problem, verbose=1)

    controller = RLController(config={
        'experiment': '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/unique_training_data_2022_10_11_20_24_42/',
        'checkpoint': 255,
        'missions': {
            'folder': '/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/verification_1000_problems/',
        },
    }, problem=problem, arena=arena)

    observation = arena.reset(problem.start_state)
    for i in tqdm(range(100)):
        action = controller.get_action(observation)
        observation = arena.step(action)

    arena.plot_all_on_map(problem=problem)