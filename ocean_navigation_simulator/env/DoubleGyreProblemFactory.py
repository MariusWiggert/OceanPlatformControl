import abc
import datetime as dt
from typing import Optional

import numpy as np

from ocean_navigation_simulator.env.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.env.utils import units


class DoubleGyreProblemFactory(ProblemFactory):
    def __init__(self, seed: Optional[float] = 2021):
        self.rng = np.random.default_rng(seed)

    def next_problem(self) -> NavigationProblem:
        length = np.sqrt(self.rng.uniform(0, 1))
        angle = np.pi * self.rng.uniform(0, 2)

        start_state = PlatformState(
            lon=units.Distance(deg=1.5+0.25 * length * np.cos(angle)),
            lat=units.Distance(deg=0.5+0.25 * length * np.sin(angle)),
            date_time=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
        )

        length = np.sqrt(self.rng.uniform(0, 1))
        angle = np.pi * self.rng.uniform(0, 2)

        end_region = SpatialPoint(
            lon=units.Distance(deg=0.5+0.25 * length * np.cos(angle)),
            lat=units.Distance(deg=0.5+0.25 * length * np.sin(angle)),
        )
        target_radius = 1/50

        return NavigationProblem(
            start_state=start_state,
            end_region=end_region,
            target_radius=target_radius,
        )