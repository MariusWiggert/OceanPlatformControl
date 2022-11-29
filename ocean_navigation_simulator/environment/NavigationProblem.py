import dataclasses
import datetime
from typing import List, Optional, Union

import matplotlib
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    PlatformStateSet,
    SpatialPoint,
)
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import get_markers

# TODO: add minimal docstrings and comments for others to build on it!


@dataclasses.dataclass
class NavigationProblem(Problem):
    start_state: Union[PlatformState, PlatformStateSet]
    end_region: SpatialPoint
    target_radius: float
    timeout: datetime.timedelta = None
    platform_dict: dict = None
    x_range: List = None
    y_range: List = None
    extra_info: dict = None

    def __post_init__(self):
        if type(self.start_state) is PlatformStateSet:
            self.nb_platforms = len(self.start_state)
        else:
            self.nb_platforms = 1

    def passed_seconds(self, state: PlatformState, id_to_comp: Optional[int] = 0) -> float:
        if self.nb_platforms == 1:
            return (state.date_time - self.start_state.date_time).total_seconds()
        else:
            return (state.date_time - self.start_state[id_to_comp].date_time).total_seconds()

    def distance(self, state: PlatformState) -> float:
        return self.end_region.distance(state.to_spatial_point())

    def angle(self, state: PlatformState) -> float:
        return self.end_region.angle(state.to_spatial_point())

    def is_done(self, state: PlatformState) -> int:
        if self.passed_seconds(state) >= self.timeout.total_seconds():
            return -1
        elif state.distance(self.end_region).deg <= self.target_radius:
            return 1
        return 0

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        problem_start_color: Optional[str] = "red",
        problem_target_color: Optional[str] = "green",
    ) -> matplotlib.axes.Axes:

        markers = get_markers()

        if self.nb_platforms > 1:
            for lon, lat, id in zip(
                self.start_state.lon.deg, self.start_state.lat.deg, range(self.nb_platforms)
            ):
                ax.scatter(
                    lon,
                    lat,
                    c=problem_start_color,
                    marker=next(markers),
                    label=f"start platform {id}",
                    s=150,
                )
        else:
            ax.scatter(
                self.start_state.lon.deg,
                self.start_state.lat.deg,
                c=problem_start_color,
                marker="o",
                label="start platform",
                s=150,
            )

        ax.add_patch(
            plt.Circle(
                (self.end_region.lon.deg, self.end_region.lat.deg),
                self.target_radius,
                facecolor=problem_target_color,
                edgecolor=problem_target_color,
                label="goal",
            )
        )

        return ax

    @staticmethod
    def from_pandas_row(mission):
        return NavigationProblem(
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
            x_range=[
                units.Distance(deg=mission["x_range_l"]),
                units.Distance(deg=mission["x_range_h"]),
            ]
            if "x_range_l" in mission
            else None,
            y_range=[
                units.Distance(deg=mission["y_range_l"]),
                units.Distance(deg=mission["y_range_h"]),
            ]
            if "x_range_h" in mission
            else None,
            extra_info=mission.to_dict() | {"index": mission.name},
        )

    def to_dict(self, pltf_id_for_timeref: Optional[int] = 0) -> dict:
        return (
            {
                "t_0": self.start_state.date_time.isoformat()
                if self.nb_platforms == 1
                else self.start_state[pltf_id_for_timeref].date_time.isoformat(),
                "x_0_lon": self.start_state.lon.deg,
                "x_0_lat": self.start_state.lat.deg,
                "x_T_lon": self.end_region.lon.deg,
                "x_T_lat": self.end_region.lat.deg,
                "target_radius": self.target_radius,
                "timeout_in_h": self.timeout.total_seconds() / 3600,
            }
            | (
                {
                    "x_range_l": self.x_range[0].deg,
                    "x_range_h": self.x_range[1].deg,
                }
                if self.x_range is not None
                else {}
            )
            | (
                {
                    "x_range_l": self.y_range[0].deg,
                    "x_range_h": self.y_range[1].deg,
                }
                if self.y_range is not None
                else {}
            )
            | (self.extra_info if self.extra_info is not None else {})
        )

    def __repr__(self):
        return "Problem [start: {s}, end: {e}, optimal time: {ot:.1f}, timeout: {t:.1f}h".format(
            s=self.start_state.to_spatio_temporal_point(),
            e=self.end_region,
            ot=self.extra_info["optimal_time_in_h"]
            if "optimal_time_in_h" in self.extra_info
            else float("inf"),
            t=self.timeout.total_seconds() / 3600,
        )
