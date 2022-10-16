#%%
import pickle

import numpy as np
import ray.rllib.utils
from ray.rllib.agents.ppo import PPOTrainer
from tqdm import tqdm
import time
import os

from ocean_navigation_simulator.controllers.RLController import RLControllerFromAgent
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.reinforcement_learning.DoubleGyreFeatureConstructor import DoubleGyreFeatureConstructor
from ocean_navigation_simulator.problem_factories.DoubleGyreProblemFactory import DoubleGyreProblemFactory
from ocean_navigation_simulator.reinforcement_learning.DoubleGyreEnv import DoubleGyreEnv
from scripts.jerome.old import clean_ray_results
from ocean_navigation_simulator.utils.plotting_utils import get_index_from_posix_time

script_start_time = time.time()

rng = np.random.default_rng(2022)

def env_creator(env_config):
    return DoubleGyreEnv(config={
        'seed': env_config.worker_index * (env_config.vector_index+1),
        'arena_steps_per_env_step': 1,
        'scenario_name': 'double_gyre',
    })

ray.tune.registry.register_env("PlatformEnv", env_creator)

factory = DoubleGyreProblemFactory(scenario_name='simplified', seed=2022)

episode = 400
model_name = 'angle_feature_simple_nn_with_currents'
#model_name = 'last'

experiment_path = 'models/simplified_double_gyre/'
experiments = os.listdir(experiment_path)
experiments = [x for x in experiments if not x.startswith('.')]
experiments = [os.path.join(experiment_path, f) for f in experiments]
experiments.sort(key=lambda x: os.path.getmtime(x))
last = experiments[-1]
model_path = last+'/' if model_name == 'last' else experiment_path+model_name+'/'

config = pickle.load(open(model_path+'config.p', "rb"))
config['num_workers'] = 1
config['explore'] = False
config["in_evaluation"] = True,
agent = PPOTrainer(config=config)
agent.restore(model_path + f'checkpoints/checkpoint_{episode:06d}/checkpoint-{episode}')
controller = RLControllerFromAgent(problem=factory.next_problem(), agent=agent, feature_constructor=DoubleGyreFeatureConstructor())

#
# tf_model = agent.get_policy().model.base_model
# tf_model(tf.tensor([]))

arenas = []
problems = []
success = []

for j in tqdm(range(100)):
    problem = factory.next_problem()
    # controller = NaiveToTargetController(problem=problem)
    problems.append(problem)

    arena = ArenaFactory.create(scenario_name='double_gyre')
    #arena = ArenaFactory.create(scenario_name='current_highway')
    observation = arena.reset(problem.start_state)
    arenas.append(arena)

    controller.problem = problem
    solved = False

    for i in range(1000): # 1000 * 0.02s = 200s
        action = controller.get_action(observation)
        observation = arena.step(action)
        prolem_status = problem.is_done(observation.platform_state)
        if prolem_status == 1:
            solved = True
            break
        if prolem_status == -1 or not arena.is_inside_arena():
            break

    success.append(solved)

success_rate = sum([1 if s else 0 for s in success]) / len(success)
print(f'Success Rate: {success_rate:.1%}')

plot_folder=model_path+'plots/'
import os
if not os.path.isdir(plot_folder):
    os.mkdir(plot_folder)

# Control/Battery/Seaweed
# fig, axs = plt.subplots(2, 2)
# for i, arena in enumerate(arenas):
#     arena.plot_battery_trajectory_on_timeaxis(ax=axs[0,0])
#     arena.plot_seaweed_trajectory_on_timeaxis(ax=axs[0,1])
#     arena.plot_control_thrust_on_timeaxis(ax=axs[1,0])
#     arena.plot_control_angle_on_timeaxis(ax=axs[1,1])
# plt.tight_layout()
# fig.show()
# fig.savefig(plot_folder+f'{episode}_battery.pdf')

def add_colored_trajecotry_and_problem(ax, posix_time):
    for i, arena in enumerate(arenas):
        index = get_index_from_posix_time(posix_time)
        color = 'green' if success[i] else 'red'
        arena.plot_all_on_map(
            ax=ax,
            index=index,
            current_position_color=color,
            state_color=color,
            control_stride=5,
            problem=problems[i],
            problem_color=color,
        )

# Static
ax = arenas[0].ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=0,
    x_interval = arena.spatial_boundary['x'],
    y_interval = arena.spatial_boundary['y'],
    return_ax=True,
)
add_colored_trajecotry_and_problem(ax, 0)
ax.get_figure().show()
ax.get_figure().savefig(plot_folder+f'{episode}_static.pdf')

# Animation
arenas[0].ocean_field.hindcast_data_source.animate_data(
    x_interval=[-0.2,2.2],
    y_interval=[-0.1,1.1],
    t_interval=[0,20],
    temporal_resolution=0.1,
    spatial_resolution=0.1,
    output='safari',
    add_ax_func=add_colored_trajecotry_and_problem,
)

clean_ray_results.evaluation_run(iteration_limit=10, delete=True, filter_string='PPOTrainer', ignore_last=False)


print(f"Total Script Time: {time.time()-script_start_time:.2f}s = {(time.time()-script_start_time)/60:.2f}min")

