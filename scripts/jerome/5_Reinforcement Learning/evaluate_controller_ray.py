import os.path
import sys

import pandas as pd
import ray
import time
from datetime import datetime

class Logger(object):
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def __del__(self):
        self.file.close()

results_folder = f'results/evaluate_controller_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
os.mkdir(results_folder)
# sys.stdout = Logger(results_folder+'/stdout.log')

print('Script started ...')
script_start_time = time.time()

# 1.    ray up setup/ray-config.yaml
#       ray up --restart-only setup/ray-config.yaml
#       ray dashboard setup/ray-config.yaml
#       ray attach setup/ray-config.yaml -p 10001
# 2.    ray monitor setup/ray-config.yaml
# 3.    ray submit setup/ray-config.yaml scripts/jerome/3_Cluster_RL/evaluate_controller_ray.py

# tensorboard --logdir ~/ray_results
# ssh -L 16006:127.0.0.1:6006 olivier@my_server_ip

# Documentation: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
# ray.init(address='auto')
# ray.init("ray://13.68.187.126:10001")
ray.init(
    'ray://localhost:10001',
    runtime_env={
        'working_dir': '.',
        'excludes': ['./data', './generated_media', './hj_reachability', './models', '.git', './ocean_navigation_simulator', './results'],
        'py_modules': ['ocean_navigation_simulator'],
    },
)
# ray.init()
print(f"Code sent in {time.time()-script_start_time:.1f}s")

active_nodes = list(filter(lambda node: node['Alive'] == True, ray.nodes()))
cpu_resources = ray.cluster_resources()['CPU'] if 'CPU' in ray.cluster_resources() else 0
print(f'''This cluster consists of
    {len(active_nodes)} nodes in total
    {cpu_resources} CPU resources in total''')

@ray.remote(num_cpus=1, num_gpus=1)
def evaluate(index, mission):
    import os, psutil

    from ocean_navigation_simulator.controllers.NaiveController import NaiveController
    from ocean_navigation_simulator.controllers.HJController import HJController
    from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
    from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem

    TIMING = True
    DEBUG = True

    if DEBUG:
        print(f'##### Starting Mission {index:03d} ######')
    mission_start_time = time.time()
    problem = NavigationProblem.from_mission(mission)

    # Step 1: Create Arena
    start = time.time()
    arena = ArenaFactory.create(scenario_name='gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast', verbose=True)
    observation = arena.reset(platform_state=problem.start_state)
    if TIMING:
        print(f'## Create Arena ({time.time() - start:.1f}s) ##')

    # Step 2: Create Controller
    start = time.time()
    # controller = NaiveToTargetController(problem=problem)
    controller = HJController(problem=problem, platform_dict=arena.platform.platform_dict)
    if TIMING:
        print(f'## Create Controller ({time.time() - start:.1f}s) ##')

    # Step 3: Running Arena
    start = time.time()
    steps = 0
    while True:
        action = controller.get_action(observation)
        observation = arena.step(action)
        problem_status = arena.problem_status(problem=problem)
        steps += 1

        # Simulation Termination
        if problem_status == 1 or problem_status == -1:
            break
    if TIMING:
        print(f'## Running Arena ({time.time() - start:.1f}s) ##')

    result = {
        'index': index,
        'controller': type(controller).__name__,
        'success': True if problem_status==1 else False,
        'steps': steps,
        'running_time': problem.passed_seconds(observation.platform_state),
        'final_distance': problem.distance(observation.platform_state),
        'pid': os.getpid(),
        'process_time': time.time() - mission_start_time,
        'process_memory': psutil.Process().memory_info().rss,
    }

    if DEBUG:
        print(f'##### Finished (Mission {index:03d}: {"Success" if result["success"] else "Failure"}, {steps} Steps, {result["running_time"]/(3600*24):.0f}d {result["running_time"] % (3600*24) / 3600:.0f}h, {result["final_distance"]:.4f} Degree) (Process: {result["process_time"]:.1f}s, {result["process_memory"] / 1e6:.1f}MB)')

    return result

feasible_df = pd.read_csv('./data/missions/validation/feasible.csv', index_col=0)
ray_results = ray.get([evaluate.remote(index, row) for index, row in feasible_df.iterrows()])

results_df = pd.DataFrame(ray_results).set_index('index').rename_axis(None)
results_df.to_csv(f'{results_folder}/results.csv')

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/60:.0f}min {script_time%60:.0f}s.")