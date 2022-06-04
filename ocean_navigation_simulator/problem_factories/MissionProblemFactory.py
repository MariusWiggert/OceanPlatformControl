import datetime as dt
from typing import Optional, List
import pandas as pd

from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.problem_factories.ProblemFactory import ProblemFactory
from ocean_navigation_simulator.utils import units


class MissionProblemFactory(ProblemFactory):
    def __init__(
        self,
        scenario_name: Optional[str] = 'gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast',
        available_missions: Optional[List] = list(range(120)),
    ):
        self.scenario_name = scenario_name
        self.available_missions = available_missions
        self.mission_df = pd.read_csv('data/value_function_learning/missions.csv', index_col=0)

    def next_problem(self) -> NavigationProblem:
        if not self.has_problems_remaining():
            raise Exception("No more available Problems.")

        index = self.available_missions.pop(0)
        row = self.mission_df.iloc[index]

        return NavigationProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=row['x_0_lon']),
                lat=units.Distance(deg=row['x_0_lat']),
                date_time=dt.datetime.fromisoformat(row['t_0'])
            ),
            end_region=SpatialPoint(
                lon=units.Distance(deg=row['x_T_lon']),
                lat=units.Distance(deg=row['x_T_lat'])
            ),
            target_radius=0.1,
            timeout=100 * 3600
        )

    def has_problems_remaining(self) -> bool:
        return len(self.available_missions) > 0