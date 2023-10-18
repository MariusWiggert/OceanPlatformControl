#%%
import logging
import os

import yaml

from ocean_navigation_simulator.problem_factories.SeaweedMissionGenerator import (
    SeaweedMissionGeneratorFeasibility,
)
from ocean_navigation_simulator.utils.misc import set_arena_loggers

os.chdir(
    "/Users/matthiaskiller/Library/Mobile Documents/com~apple~CloudDocs/Studium/Master RCI/Masters Thesis/Code/OceanPlatformControl"
)

set_arena_loggers(logging.INFO)

with open("config/arena/seaweed_mission_generation.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("config/arena/" + config["scenario_name"] + ".yaml") as f:
    arenaConfig = yaml.load(f, Loader=yaml.FullLoader)

config.update({"platform_dict": arenaConfig["platform_dict"]})

#%%
generator = SeaweedMissionGeneratorFeasibility(config=config)
# %%
problem_batch = generator.generate_batch()

mission_conf = [problem.to_c3_mission_config() for problem in problem_batch]

# %%
generator.plot_starts(config=config, problems=problem_batch)
