import abc
import datetime as dt
from typing import Optional

import numpy as np

from ocean_navigation_simulator.env.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.env.utils import units


class CurrentHighwayProblemFactory(ProblemFactory):
    def __init__(self, seed: Optional[float] = 2021):
        self.rng = np.random.default_rng(seed)

    def next_problem(self) -> NavigationProblem:
        x = self.rng.uniform(0, 10)
        y = 2 #self.rng.uniform(0, 5)

        start_state = PlatformState(
            lon=units.Distance(deg=x),
            lat=units.Distance(deg=y),
            date_time=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
        )

        x = self.rng.uniform(0, 10)
        y = 8 #self.rng.uniform(5, 10)

        end_region = SpatialPoint(
            lon=units.Distance(deg=x),
            lat=units.Distance(deg=y),
        )
        target_radius = 10/50

        return NavigationProblem(
            start_state=start_state,
            end_region=end_region,
            target_radius=target_radius,
        )