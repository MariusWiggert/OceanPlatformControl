import dataclasses
import datetime
from typing import Optional, List

import matplotlib
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.utils import units


@dataclasses.dataclass
class NavigationProblem(Problem):
    start_state: PlatformState
    end_region: SpatialPoint
    target_radius: float
    timeout: datetime.timedelta = None
    platform_dict: dict = None
    optimal_time: datetime.timedelta = None
    x_range: List = None
    y_range: List = None
    extra_info: dict = None

    def passed_seconds(self, state: PlatformState) -> float:
        return (state.date_time - self.start_state.date_time).total_seconds()

    def distance(self, state: PlatformState) -> float:
        return self.end_region.distance(state.to_spatial_point())

    def is_done(self, state: PlatformState) -> int:
        if self.passed_seconds(state) >= self.timeout.total_seconds():
            return -1
        elif state.distance(self.end_region) <= self.target_radius:
            return 1
        return 0

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        color: Optional[str] = 'green',
    ) -> matplotlib.axes.Axes:
        ax.scatter(self.start_state.lon.deg, self.start_state.lat.deg, facecolors='none', edgecolors=color, marker='o', label='start')
        ax.add_patch(plt.Circle((self.end_region.lon.deg, self.end_region.lat.deg), self.target_radius, facecolor='none', edgecolor=color, label='goal'))

        return ax

    @staticmethod
    def from_dict(mission, extra_info=None):
        return NavigationProblem(
            start_state=PlatformState(
                lon=units.Distance(deg=mission['x_0_lon']),
                lat=units.Distance(deg=mission['x_0_lat']),
                date_time=datetime.datetime.fromisoformat(mission['t_0'])
            ),
            end_region = SpatialPoint(
                lon=units.Distance(deg=mission['x_T_lon']),
                lat=units.Distance(deg=mission['x_T_lat'])
            ),
            target_radius=mission['target_radius'],
            timeout=datetime.timedelta(hours=mission['timeout_in_h']),
            x_range=[units.Distance(deg=mission['x_range_l']), units.Distance(deg=mission['x_range_h'])] if 'x_range_l' in mission else None,
            y_range=[units.Distance(deg=mission['y_range_l']), units.Distance(deg=mission['y_range_h'])] if 'x_range_h' in mission else None,
            extra_info=extra_info if extra_info is not None else {},
        )

    def to_dict(self) -> dict:
        return {
            't_0': self.start_state.date_time.isoformat(),
            'x_0_lon': self.start_state.lon.deg,
            'x_0_lat': self.start_state.lat.deg,
            'x_T_lon': self.end_region.lon.deg,
            'x_T_lat': self.end_region.lat.deg,
            'target_radius': self.target_radius,
            'timeout_in_h': self.timeout.total_seconds() / 3600,
            'optimal_time': self.optimal_time,
        } | ({
            'x_range_l': self.x_range[0].deg,
            'x_range_h': self.x_range[1].deg,
        } if self.x_range is not None else {}) | ({
            'x_range_l': self.y_range[0].deg,
            'x_range_h': self.y_range[1].deg,
        } if self.y_range is not None else {}) | (self.extra_info if self.extra_info is not None else {})

    def __repr__(self):
        return f'Problem [start: {self.start_state.to_spatial_point()}, end: {self.end_region}, optimal time: {self.optimal_time}'
