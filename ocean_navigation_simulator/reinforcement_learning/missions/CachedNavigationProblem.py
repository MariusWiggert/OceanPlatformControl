import dataclasses
import datetime
import json

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
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
    def from_pandas_row(mission, base_path=None):
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
        }

    def to_c3_mission_config(self):
        """To easily populate c3 database with missions."""
        prob_dict = self.to_dict()
        x_0 = {
            "lon": prob_dict['x_0_lon'].tolist(),
            "lat": prob_dict['x_0_lat'].tolist(),
            "date_time": prob_dict['t_0'],
        }

        x_T = {"lon": prob_dict['x_T_lon'], "lat": prob_dict['x_T_lat']}

        mission_config = {
            "x_0": [x_0], # Evaluation runner assumes it is a list (for multi-agent)
            "x_T": x_T,
            "target_radius": prob_dict['target_radius'],
            "seed": prob_dict.get('factory_seed', None),
            "feasible": prob_dict.get('feasible', None),
            'ttr_in_h': prob_dict.get('ttr_in_h', None)
        }

        return mission_config


    def get_cached_forecast_planner(self, base_path, arena=None, pickle=False):
        if pickle:
            planner = HJReach2DPlanner.from_pickle(base_path + "forecast_planner_idx_0.p")
        else:
            planner = HJReach2DPlanner.from_saved_planner_state(
                folder=f'{base_path}groups/group_{self.extra_info["group"]}/batch_{self.extra_info["batch"]}/forecast_planner_idx_0/',
                problem=self,
                specific_settings={
                    "load_plan": True,
                    "planner_path": f'{base_path}groups/group_{self.extra_info["group"]}/batch_{self.extra_info["batch"]}/',
                },
            )
        if arena is not None:
            planner.last_data_source = arena.ocean_field.forecast_data_source
        return planner

    def get_cached_hindcast_planner(self, base_path, arena=None, pickle=False):
        if pickle:
            planner = HJReach2DPlanner.from_pickle(base_path + "hindcast_planner.p")
        else:
            planner = HJReach2DPlanner.from_saved_planner_state(
                folder=f'{base_path}groups/group_{self.extra_info["group"]}/batch_{self.extra_info["batch"]}/hindcast_planner/',
                problem=self,
                specific_settings={
                    "load_plan": False,
                },
            )
        if arena is not None:
            planner.last_data_source = arena.ocean_field.hindcast_data_source
        return planner

    def __repr__(self):
        return "Problem [start: {start}, end: {end}, target_radius: {r:.2f}, ttr: {opt:.0f}h] (I{i}, G{gr} B{b} FI{fi})".format(
            start=self.start_state.to_spatio_temporal_point(),
            end=self.end_region,
            r=self.target_radius,
            opt=self.extra_info.get("ttr_in_h", float("Inf")),
            i=self.extra_info.get("index", "None"),
            gr=self.extra_info.get("group", "None"),
            b=self.extra_info.get("batch", "None"),
            fi=self.extra_info.get("factory_index", "None"),
        )
