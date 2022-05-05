import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.CurrentHighwayProblemFactory import \
    CurrentHighwayProblemFactory
from ocean_navigation_simulator.env.DoubleGyreProblemFactory import DoubleGyreProblemFactory
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController


start = time.time()

#factory = DoubleGyreProblemFactory()
factory = CurrentHighwayProblemFactory()
arenas = []
problems = []
success = []

for j in tqdm(range(50)):
    problem = factory.next_problem()
    problems.append(problem)

    #arena = ArenaFactory.create(scenario_name='double_gyre')
    arena = ArenaFactory.create(scenario_name='current_highway')
    observation = arena.reset(problem.start_state)
    arenas.append(arena)

    controller = NaiveToTargetController(problem=problem)
    is_done = False

    for i in range(1000): # 1000 * 0.02s = 20s
        action = controller.get_action(observation)
        observation = arena.step(action)
        if problem.is_done(observation.platform_state):
            is_done = True
            break
        if not arena.is_inside_arena():
            break

    success.append(is_done)

success_rate = sum([1 if s else 0 for s in success]) / len(success)
print(f'Success Rate: {success_rate:.1%}')

# Control/Battery/Seaweed
fig, axs = plt.subplots(2, 2)
for i, arena in enumerate(arenas):
    axs[0,0] = arena.plot_battery_trajectory_on_timeaxis(ax=axs[0,0])
    axs[0,1] = arena.plot_seaweed_trajectory_on_timeaxis(ax=axs[0,1])
    axs[1,0] = arena.plot_control_trajectory_on_timeaxis(ax=axs[1,0])
plt.tight_layout()
fig.show()

def add_colored_trajecotry_and_problem(ax, posix_time):
    for i, arena in enumerate(arenas):
        index = arena.get_index_from_posix_time(posix_time)
        color = 'green' if success[i] else 'red'

        ax = arena.plot_state_trajectory_on_map(
            ax=ax,
            color=color,
        )
        if not success[i]:
            ax = arena.plot_control_trajectory_on_map(
                ax=ax,
                stride=20,
            )
        ax = arena.plot_current_position_on_map(
            index=index,
            ax=ax,
            color=color,
        )
        ax = problems[i].plot(
            ax=ax,
            color=color,
        )
    return ax

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

