import dataclasses
import datetime

import json

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.utils import units


@dataclasses.dataclass
class CachedNavigationProblem(NavigationProblem):
    """extension of NavigationProblem for problems with cached planners"""

    extra_info: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_pandas_row(mission):
        def mutate_read(k, v):
            if k in ["x_cache", "y_cache"]:
                v = json.loads(v)
                v = [units.Distance(deg=i) for i in v]
            if k in ["timeout_datetime"]:
                v = datetime.datetime.fromisoformat(v)

            return v

        return CachedNavigationProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=mission["x_0_lon"]),
                lat=units.Distance(deg=mission["x_0_lat"]),
                date_time=datetime.datetime.fromisoformat(mission["t_0"]),
            ),
            end_region=SpatialPoint(
                lon=units.Distance(deg=mission["x_T_lon"]),
                lat=units.Distance(deg=mission["x_T_lat"]),
            ),
            target_radius=mission["target_radius"],
            timeout=datetime.timedelta(hours=mission["timeout_in_h"]),
            extra_info={k: mutate_read(k, v) for k, v in mission.to_dict().items()},
        )

    def to_dict(self) -> dict:
        def mutate_write(k, v):
            return v

        return {k: mutate_write(k, v) for k, v in self.extra_info.items()} | {
            "t_0": self.start_state.date_time.isoformat(),
            "x_0_lon": self.start_state.lon.deg,
            "x_0_lat": self.start_state.lat.deg,
            "x_T_lon": self.end_region.lon.deg,
            "x_T_lat": self.end_region.lat.deg,
            "target_radius": self.target_radius,
            "timeout_in_h": self.timeout.total_seconds() / 3600,
        }

    def get_cached_forecast_planner(self, base_path):
        return HJReach2DPlanner.from_saved_planner_state(
            folder=f'{base_path}groups/group_{self.extra_info["group"]}/batch_{self.extra_info["batch"]}/forecast_planner_idx_0/',
            problem=self,
            specific_settings={
                "load_plan": True,
                "planner_path": f'{base_path}groups/group_{self.extra_info["group"]}/batch_{self.extra_info["batch"]}/',
            },
        )

    def get_cached_hindcast_planner(self, base_path):
        return HJReach2DPlanner.from_saved_planner_state(
            folder=f'{base_path}groups/group_{self.extra_info["group"]}/batch_{self.extra_info["batch"]}/hindcast_planner/',
            problem=self,
            specific_settings={
                "load_plan": False,
            },
        )

    def __repr__(self):
        return "Problem [start: {start}, end: {end}, target_radius: {r:.2f}, ttr: {opt:.0f}h, timeout: {t:.0f}h]".format(
            start=self.start_state.to_spatio_temporal_point(),
            end=self.end_region,
            r=self.target_radius,
            opt=self.extra_info.get("ttr_in_h", float("Inf")),
            t=self.timeout.total_seconds() / 3600,
        )
