import dataclasses
import datetime

from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint, PlatformState
from ocean_navigation_simulator.utils import units


@dataclasses.dataclass
class CachedNavigationProblem(NavigationProblem):
    """extension of NavigationProblem for problems with cached planners"""

    extra_info: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_pandas_row(mission):
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
            extra_info=mission.to_dict() | {"index": mission.name},
        )

    def to_dict(self) -> dict:
        return {
            "t_0": self.start_state.date_time.isoformat(),
            "x_0_lon": self.start_state.lon.deg,
            "x_0_lat": self.start_state.lat.deg,
            "x_T_lon": self.end_region.lon.deg,
            "x_T_lat": self.end_region.lat.deg,
            "target_radius": self.target_radius,
            "timeout_in_h": self.timeout.total_seconds() / 3600,
        } | (self.extra_info if self.extra_info is not None else {})

    def __repr__(self):
        return "Problem [start: {start}, end: {end}, target_radius: {r:.2f}, ttr: {opt:.0f}h, timeout: {t:.1f}h]".format(
            start=self.start_state.to_spatio_temporal_point(),
            end=self.end_region,
            r=self.target_radius,
            opt=self.extra_info.get("ttr_in_h", float("Inf")),
            t=self.timeout.total_seconds() / 3600,
        )
