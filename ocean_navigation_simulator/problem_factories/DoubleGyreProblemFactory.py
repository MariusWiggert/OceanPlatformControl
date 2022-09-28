import datetime as dt
from typing import Optional

import numpy as np

from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.environment.ProblemFactory import (
    ProblemFactory,
)
from ocean_navigation_simulator.utils import units


class DoubleGyreProblemFactory(ProblemFactory):
    def __init__(self, seed: Optional[float] = 2022, scenario_name: Optional[str] = "paper"):
        self.rng = np.random.default_rng(seed)
        self.scenario_name = scenario_name

        # print(f'Problem Factory initialized with seed {seed}')

    def next_problem(self) -> NavigationProblem:
        if self.scenario_name == "paper":
            length = np.sqrt(self.rng.uniform(0, 1))
            angle = np.pi * self.rng.uniform(0, 2)

            start_x = 1.5 + 0.25 * length * np.cos(angle)
            start_y = 0.5 + 0.25 * length * np.sin(angle)

            length = np.sqrt(self.rng.uniform(0, 1))
            angle = np.pi * self.rng.uniform(0, 2)

            goal_x = 0.5 + 0.25 * length * np.cos(angle)
            goal_y = 0.5 + 0.25 * length * np.sin(angle)
        elif self.scenario_name == "simplified":
            start_x = self.rng.uniform(0, 2)
            start_y = self.rng.uniform(0, 1)

            goal_x = 0.5
            goal_y = 0.5
        else:
            raise NotImplementedError(f"Scenario {self.scenario_name} not implemented!")

        return NavigationProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=start_x),
                lat=units.Distance(deg=start_y),
                date_time=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
            ),
            end_region=SpatialPoint(
                lon=units.Distance(deg=goal_x),
                lat=units.Distance(deg=goal_y),
            ),
            target_radius=1 / 50,
            timeout=200,
        )

    def has_problems_remaining(self) -> bool:
        return True
