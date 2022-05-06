#%%

import matplotlib.pyplot as plt
from ray.rllib.agents.ppo import PPOTrainer
import ray.rllib.utils
from tqdm import tqdm
import time
import pickle


import gym

from ocean_navigation_simulator.env.Arena import Arena
from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.CurrentHighwayProblemFactory import \
    CurrentHighwayProblemFactory
from ocean_navigation_simulator.env.DoubleGyreProblemFactory import DoubleGyreProblemFactory
from ocean_navigation_simulator.env.FeatureConstructors import \
    double_gyre_simple_feature_constructor
from ocean_navigation_simulator.env.PlatformEnv import PlatformEnv
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.controllers.RLControllerFromAgent import RLControllerFromAgent

start = time.time()

gym.envs.register(
    id='DoubleGyre-v0',
    entry_point='ocean_navigation_simulator.env.PlatformEnv:PlatformEnv',
    kwargs={
        'seed': 2022,
        'env_steps_per_arena_steps': 1,
    },
    max_episode_steps=1000,
)
env = gym.make('DoubleGyre-v0')
ray.tune.registry.register_env("DoubleGyre-v0", lambda config: PlatformEnv())

factory = DoubleGyreProblemFactory(scenario_name='simplified')
#factory = CurrentHighwayProblemFactory(scenar)
arenas = []
problems = []
success = []

# controller = NaiveToTargetController(problem=problem)
config = pickle.load(open("ocean_navigation_simulator/models/simplified_double_gyre/config.p", "rb"))
agent = PPOTrainer(config=config)
agent.restore('ocean_navigation_simulator/models/simplified_double_gyre/checkpoint_000020/checkpoint-20')
controller = RLControllerFromAgent(problem=factory.next_problem(), agent=agent, feature_constructor=double_gyre_simple_feature_constructor)

for j in tqdm(range(5)):
    problem = factory.next_problem()
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

# Control/Battery/Seaweed
fig, axs = plt.subplots(2, 2)
for i, arena in enumerate(arenas):
    arena.plot_battery_trajectory_on_timeaxis(ax=axs[0,0])
    arena.plot_seaweed_trajectory_on_timeaxis(ax=axs[0,1])
    arena.plot_control_thrust_on_timeaxis(ax=axs[1,0])
    arena.plot_control_angle_on_timeaxis(ax=axs[1,1])
plt.tight_layout()
fig.show()

def add_colored_trajecotry_and_problem(ax, posix_time):
    for i, arena in enumerate(arenas):
        index = arena.get_index_from_posix_time(posix_time)
        color = 'green' if success[i] else 'red'
        arena.plot_all_on_map(
            ax=ax,
            index=index,
            current_position_color=color,
            state_color=color,
            control_stride=50,
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

# Animation
# arenas[0].ocean_field.hindcast_data_source.animate_data(
#     x_interval=[-0.2,2.2],
#     y_interval=[-0.1,1.1],
#     t_interval=[0,20],
#     temporal_res=0.1,
#     spatial_res=0.1,
#     output='safari',
#     add_ax_func=add_colored_trajecotry_and_problem,
# )


print("Total Script Time: ", time.time() - start)

