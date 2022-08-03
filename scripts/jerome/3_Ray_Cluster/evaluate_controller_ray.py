import pandas as pd
import ray
import time
from datetime import datetime

print('Script started ...')
script_start_time = time.time()

# ray.init(address='auto')
# ray.init("ray://13.68.187.126:10001")
# Documentation: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
# ray.init(
#     'ray://localhost:10001',
#     runtime_env={
#         'working_dir': '.',
#         'excludes': ['data', 'generated_media', 'hj_reachability', 'models', '.git', 'ocean_navigation_simulator', 'results'],
#         'py_modules': ['ocean_navigation_simulator'],
#     },
# )
ray.init()
print(f"Code sent in {time.time()-script_start_time:.1f}s")
active_nodes = list(filter(lambda node: node['Alive'] == True, ray.nodes()))
cpu_resources = ray.cluster_resources()['CPU'] if 'CPU' in ray.cluster_resources() else 0
print(f'''This cluster consists of
    {len(active_nodes)} nodes in total
    {cpu_resources} CPU resources in total''')

@ray.remote(num_cpus=8)
def evaluate(index, mission):
    import os, psutil
    from requests import get
    import socket
    from c3python import C3Python # https://github.com/c3aidti/c3python

    from ocean_navigation_simulator.controllers.NaiveToTargetController import NaiveToTargetController
    from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
    from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem

    private_ip = socket.gethostbyname(socket.gethostname())
    public_ip = get('https://api.ipify.org').content.decode('utf8')
    pid = os.getpid()

    TIMING = True
    DEBUG = True

    if DEBUG:
        print(f'##### Starting Mission {index:03d} ######')
    mission_start_time = time.time()
    problem = NavigationProblem.from_mission(mission)

    # Step 1: Download Files
    start = time.time()
    ArenaFactory.download_files(problem=problem, n_days_ahead=6)
    if TIMING:
        print(f'## Download Files ({time.time() - start:.1f}s) ##')

    # Step 2: Create Controller
    start = time.time()
    controller = NaiveToTargetController(problem=problem)
    if TIMING:
        print(f'## Create Controller ({time.time() - start:.1f}s) ##')

    # Step 3: Create Arena
    start = time.time()
    # arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast', pid=pid, timing=True)
    arena = ArenaFactory.create(scenario_name='gulf_of_mexico_Copernicus_hindcast', pid=pid, timing=False)
    observation = arena.reset(platform_state=problem.start_state)
    if TIMING:
        print(f'## Create Arena ({time.time() - start:.1f}s) ##')

    # Step 4: Running Arena
    start = time.time()
    step = 1
    while True:
        action = controller.get_action(observation)
        observation = arena.step(action)

        # Simulation Termination
        problem_status = problem.is_done(observation.platform_state)
        if not arena.is_inside_arena():
            problem_status = -1
        # print(f"Step {step}: Time = {observation.platform_state.date_time}, Passed = {problem.passed_seconds(observation.platform_state):.0f}s, Action = {action.direction} rad")
        if problem_status == 1 or problem_status == -1:
            break
        step += 1

    if TIMING:
        print(f'## Running Arena ({time.time() - start:.1f}s) ##')

    process_time = time.time() - mission_start_time
    if DEBUG:
        print(f'##### Finished Mission {index:03d} with {"Sucess" if problem_status==1 else "Failure" } ({process_time:.1f}s, {psutil.Process().memory_info().rss / 1e6:.1f}MB')

    return [index, process_time, problem_status, problem.passed_seconds(observation.platform_state)]

feasible = pd.read_csv('./data/missions/feasible.csv', index_col=0)
object_ids = [evaluate.remote(index, row) for index, row in feasible.head(n=5).iterrows()]

results = ray.get(object_ids)

results_df = pd.DataFrame(results, columns = ['Index', 'Age', ], index=0)
print(results_df.head(n=5))
results_df.save('results/evaluate_controller_{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}.csv')

print(results)
print(feasible['Naive_success'])
# print(f'Success Rate: {(sum(results)/len(results) + 1) / 2:%}')

print(f"Total Script Time: {time.time()-script_start_time:.2f}s = {(time.time()-script_start_time)/60:.1f}min")