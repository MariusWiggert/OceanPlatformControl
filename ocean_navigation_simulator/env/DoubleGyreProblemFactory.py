import abc
import numpy as np

from ocean_navigation_simulator.env.DoubleGyreProblem import DoubleGyreProblem
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.env.utils import units


class DoubleGyreProblemFactory(ProblemFactory):
    def next_problem(self) -> DoubleGyreProblem:
        length = np.sqrt(np.random.uniform(0, 1))
        angle = np.pi * np.random.uniform(0, 2)

        start_state = PlatformState(
            lon=units.Distance(deg=1.5+0.25 * length * np.cos(angle)),
            lat=units.Distance(deg=0.5+0.25 * length * np.sin(angle))
        )

        length = np.sqrt(np.random.uniform(0, 1))
        angle = np.pi * np.random.uniform(0, 2)

        end_region = SpatialPoint(
            lon=units.Distance(deg=0.5+0.25 * length * np.cos(angle)),
            lat=units.Distance(deg=0.5+0.25 * length * np.sin(angle)),
        )
        radius = 1/50

        return DoubleGyreProblem(
            start_state=start_state,
            end_region=end_region,
            radius=radius,
        )