import datetime as dt

from tqdm import tqdm
import time

from ocean_navigation_simulator.controllers.ImitationController import ImitationController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.problem_factories.DoubleGyreProblemFactory import DoubleGyreProblemFactory
from ocean_navigation_simulator.utils import units

start = time.time()

arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast')

problem = NavigationProblem(
    start_state=PlatformState(
        lon=units.Distance(deg=-81.74966879179048),
        lat=units.Distance(deg=18.839454259572026),
        date_time=dt.datetime(2021, 11, 24, 12, 10, tzinfo=dt.timezone.utc)
    ),
    end_region=SpatialPoint(
        lon=units.Distance(deg=-83.17890714569597),
        lat=units.Distance(deg=18.946404547127734)
    ),
    target_radius=0.1,
    timeout=100 * 3600
)

controller = ImitationController(problem=problem, platform_dict=arena.platform.platform_dict)
observation = arena.reset(problem.start_state)


for i in tqdm(range(5000)):
    action = controller.get_action(observation)
    observation = arena.step(action)

arena.plot_all_on_map(
    problem=problem,
).get_figure().show()

print("Total Script Time: ", time.time() - start)