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
    """class to hold the essential variables for a path planning problem (A -> B)"""

    start_state: Union[PlatformState, PlatformStateSet]
    end_region: SpatialPoint
    target_radius: float
    timeout: datetime.timedelta = None
    platform_dict: dict = None
    x_range: List = None
    y_range: List = None
    extra_info: dict = dataclasses.field(default_factory=dict)

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

    def distance(self, state: PlatformState) -> units.Distance:
        return self.end_region.distance(state.to_spatial_point())

    def bearing(self, state: PlatformState) -> float:
        return self.end_region.bearing(state.to_spatial_point())

    def is_done(self, state: PlatformState) -> int:
        """
        Get the problem status
        Returns:
            1   if problem was solved
            0   if problem is still open
        """
        if state.distance(self.end_region).deg <= self.target_radius:
            return 1
        return 0

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        problem_start_color: Optional[str] = "red",  # "black",
        problem_target_color: Optional[str] = "green",
        start_size=100,
    ) -> matplotlib.axes.Axes:
        markers = get_markers()

        # if self.nb_platforms > 1:
        #     for lon, lat, id in zip(
        #         self.start_state.lon.deg, self.start_state.lat.deg, range(self.nb_platforms)
        #     ):
        #         ax.scatter(
        #             lon,
        #             lat,
        #             c=problem_start_color,
        #             marker=next(markers),
        #             label=f"start platform {id}",
        #             s=150,
        #         )
        # else:
        ax.scatter(
            self.start_state.lon.deg,
            self.start_state.lat.deg,
            c=problem_start_color,
            marker="o",
            label="start platforms",
            # facecolor="None",
            # edgecolors=problem_start_color,
            # linewidths=1,
            s=start_size if self.nb_platforms == 1 else 30,
            zorder=6,
        )

        ax.add_patch(
            plt.Circle(
                (self.end_region.lon.deg, self.end_region.lat.deg),
                self.target_radius,
                facecolor=problem_target_color,
                edgecolor=problem_target_color,
                label="goal",
                zorder=6,
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
        return "Problem [start: {s}, end: {e}, target_radius: {r:.2f}]".format(
            s=self.start_state.to_spatio_temporal_point(), e=self.end_region, r=self.target_radius
        )
