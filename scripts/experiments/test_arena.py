from ocean_navigation_simulator.env.OceanPlatformArena import OceanPlatformArena
from ocean_navigation_simulator.env.controllers.straight_line_controller import StraightLineController

arena = OceanPlatformArena(seed=1234)
controller = StraightLineController()

for i in range(100):
    action = controller(observation)
    observation = arena.step(action)



