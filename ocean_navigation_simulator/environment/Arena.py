"""A Ocean Platform Arena.
A Ocean arena contains the logic for navigating a platform in the ocean.
"""

import dataclasses
import datetime as dt
from typing import Dict, Optional, Union, Tuple

import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.Platform import Platform, PlatformAction
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint, PlatformState
from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.environment.data_sources.OceanCurrentSource.AnalyticalOceanCurrents import \
    OceanCurrentSourceAnalytical
from ocean_navigation_simulator.environment.data_sources.OceanCurrentSource.OceanCurrentSource import \
    OceanCurrentSourceXarray, OceanCurrentSource
from ocean_navigation_simulator.environment.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.environment.data_sources.SeaweedGrowthField import SeaweedGrowthField
from ocean_navigation_simulator.environment.data_sources.SolarIrradianceField import SolarIrradianceField


@dataclasses.dataclass
class ArenaObservation:
    """
    Specifies an observation from the simulator.
    This differs from SimulatorState in that the observations are not
    ground truth state, and are instead noisy observations from the
    environment.
    """
    platform_state: PlatformState  # position, time, battery
    true_current_at_state: OceanCurrentVector  # measured current at platform_state
    forecast_data_source: Union[
        OceanCurrentSource, OceanCurrentSourceXarray, OceanCurrentSourceAnalytical]  # Data Source of the forecast


class Arena:
    """A OceanPlatformArena in which an ocean platform moves through a current field."""
    ocean_field: OceanCurrentField = None
    solar_field: SolarIrradianceField = None
    seaweed_field: SeaweedGrowthField = None
    platform: Platform = None

    # TODO: where do we do the reset? I guess for us reset mostly would mean new start and goal position?
    # TODO: not sure what that should be for us, decide where to put the feature constructor
    def __init__(
            self,
            sim_cache_dict: Dict,
            platform_dict: Dict,
            ocean_dict: Dict,
            use_geographic_coordinate_system: bool,
            solar_dict: Optional[Dict] = None,
            seaweed_dict: Optional[Dict] = None,
    ):
        """OceanPlatformArena constructor.
    Args:
        sim_cache_dict:
        platform_dict:
        ocean_dict:
        use_geographic_coordinate_system: If True we use the Geographic coordinate system in lat, lon degree,
                                          if false the spatial system is in meters in x, y.
    Optional Args:
        solar_dict:
        seaweed_dict:
    """
        # Initialize the Data Fields from the respective dictionaries
        self.ocean_field = OceanCurrentField(
            sim_cache_dict=sim_cache_dict,
            hindcast_source_dict=ocean_dict['hindcast'],
            forecast_source_dict=ocean_dict['forecast'],
            use_geographic_coordinate_system=use_geographic_coordinate_system
        )

        if solar_dict is not None and solar_dict['hindcast'] is not None:
            self.solar_field = SolarIrradianceField(
                sim_cache_dict=sim_cache_dict,
                hindcast_source_dict=solar_dict['hindcast'],
                forecast_source_dict=solar_dict['forecast'],
                use_geographic_coordinate_system=use_geographic_coordinate_system
            )
        else:
            self.solar_field = None

        if seaweed_dict is not None and seaweed_dict['hindcast'] is not None:
            # For initializing the SeaweedGrowth Field we need to supply the respective SolarIrradianceSources
            seaweed_dict['hindcast']['source_settings']['solar_source'] = self.solar_field.hindcast_data_source
            if seaweed_dict['forecast'] is not None:
                seaweed_dict['forecast']['source_settings']['solar_source'] = self.solar_field.hindcast_data_source
            self.seaweed_field = SeaweedGrowthField(
                sim_cache_dict=sim_cache_dict,
                hindcast_source_dict=seaweed_dict['hindcast'],
                forecast_source_dict=seaweed_dict['forecast'],
                use_geographic_coordinate_system=use_geographic_coordinate_system
            )
        else:
            self.seaweed_field = None

        self.platform = Platform(
            platform_dict=platform_dict,
            ocean_source=self.ocean_field.hindcast_data_source,
            use_geographic_coordinate_system=use_geographic_coordinate_system,
            solar_source=self.solar_field.hindcast_data_source if self.solar_field is not None else None,
            seaweed_source=self.seaweed_field.hindcast_data_source if self.seaweed_field is not None else None
        )

        self.initial_state, self.state_trajectory, self.action_trajectory = [None] * 3

    def reset(self, platform_state: PlatformState) -> ArenaObservation:
        """Resets the arena.
        Args:
            platform_state
        Returns:
          The first observation from the newly reset simulator
        """
        self.initial_state = platform_state
        self.platform.set_state(self.initial_state)
        self.platform.initialize_dynamics(self.initial_state)
        self.ocean_field.forecast_data_source.update_casadi_dynamics(self.initial_state)

        self.state_trajectory = np.expand_dims(np.array(platform_state).squeeze(), axis=0)
        self.action_trajectory = np.zeros(shape=(0, 2))

        return ArenaObservation(platform_state=platform_state,
                                true_current_at_state=self.ocean_field.get_ground_truth(
                                    self.initial_state.to_spatio_temporal_point()),
                                forecast_data_source=self.ocean_field.forecast_data_source)

    def step(self, action: PlatformAction) -> ArenaObservation:
        """Simulates the effects of choosing the given action in the system.
        Args:
            action: The action to take in the simulator.
        Returns:
            Arena Observation including platform state, true current at platform, forecasts
        """
        state = self.platform.simulate_step(action)

        self.state_trajectory = np.append(self.state_trajectory, np.expand_dims(np.array(state).squeeze(), axis=0),
                                          axis=0)
        self.action_trajectory = np.append(self.action_trajectory, np.expand_dims(np.array(action).squeeze(), axis=0),
                                           axis=0)

        return ArenaObservation(
            platform_state=state,
            true_current_at_state=self.ocean_field.get_ground_truth(state.to_spatio_temporal_point()),
            forecast_data_source=self.ocean_field.forecast_data_source)

    def plot_control_trajectory_on_map(self, ax: Optional[matplotlib.axes.Axes] = None,
                                       color='magenta', stride: Optional[int] = 1) -> matplotlib.axes.Axes:
        """
        Plots the control trajectory (as arrows) on a spatial map. Passing in an axis is optional.
         Otherwise a new figure is created.
        Args:
            ax: Optional[matplotlib.axes.Axes]
            color: Optional[str] = 'black'
            stride: Optional[int] = 1

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        u_vec = self.action_trajectory[::stride, 0] * np.cos(self.action_trajectory[::stride, 1])
        v_vec = self.action_trajectory[::stride, 0] * np.sin(self.action_trajectory[::stride, 1])
        ax.quiver(self.state_trajectory[:-1:stride, 0], self.state_trajectory[:-1:stride, 1], u_vec, v_vec, color=color,
                  scale=15)

        return ax

    def plot_state_trajectory_on_map(self, ax: Optional[matplotlib.axes.Axes] = None,
                                     color: Optional[str] = 'black', stride: Optional[int] = 1) -> matplotlib.axes.Axes:
        """
        Plots the state trajectory on a spatial map. Passing in an axis is optional. Otherwise a new figure is created.
        Args:
            ax: Optional[matplotlib.axes.Axes]
            color: Optional[str] = 'black'
            stride: Optional[int] = 1

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.state_trajectory[::stride, 0], self.state_trajectory[::stride, 1], '-', marker='.', markersize=1,
                color=color, linewidth=1, label='State Trajectory')

        return ax

    def plot_current_position_on_map(self, index: int, ax: Optional[matplotlib.axes.Axes] = None,
                                     color: Optional[str] = 'black'):
        """
        Plots the current position at the given index on a spatial map. Passing in an axis is optional.
         Otherwise a new figure is created.
        Args:
            index: int,
            ax: Optional[matplotlib.axes.Axes]
            color: Optional[str] = 'black'

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(self.state_trajectory[index, 0], self.state_trajectory[index, 1], c=color, marker='.', s=100,
                   label='position')

        return ax

    def plot_battery_trajectory_on_timeaxis(self, ax: Optional[matplotlib.axes.Axes] = None,
                                            stride: Optional[int] = 1) -> matplotlib.axes.Axes:
        """
        Plots the battery capacity on a time axis. Passing in an axis is optional. Otherwise a new figure is created.
        Args:
            ax: Optional[matplotlib.axes.Axes]
            stride: Optional[int] = 1

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        locator = matplotlib.dates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = matplotlib.dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        dates = [dt.datetime.fromtimestamp(posix, tz=dt.timezone.utc) for posix in self.state_trajectory[::stride, 2]]
        ax.plot(dates, self.state_trajectory[::stride, 3])

        ax.set_title('Battery charge over time')
        ax.set_ylim(0., 1.1)
        ax.set_xlabel('time in h')
        ax.set_ylabel('Battery Charging level [0,1]')

        return ax

    def plot_seaweed_trajectory_on_timeaxis(self, ax: Optional[matplotlib.axes.Axes] = None,
                                            stride: Optional[int] = 1) -> matplotlib.axes.Axes:
        """
        Plots the seaweed mass on a time axis. Passing in an axis is optional. Otherwise a new figure is created.
        Args:
            ax: Optional[matplotlib.axes.Axes]
            stride: Optional[int] = 1

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        locator = matplotlib.dates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = matplotlib.dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        dates = [dt.datetime.fromtimestamp(posix, tz=dt.timezone.utc) for posix in self.state_trajectory[::stride, 2]]
        ax.plot(dates, self.state_trajectory[::stride, 3], marker='.')

        ax.set_title('Seaweed Mass over Time')
        ax.set_ylim(0., 1.1)
        ax.set_xlabel('time in h')
        ax.set_ylabel('Seaweed Mass in kg')

        return ax

    def plot_control_trajectory_on_timeaxis(self, ax: Optional[matplotlib.axes.Axes] = None,
                                            stride: Optional[int] = 1) -> matplotlib.axes.Axes:
        """
        Plots the control thrust/angle on a time axis. Passing in an axis is optional.
         Otherwise a new figure is created.
        Args:
            ax: Optional[matplotlib.axes.Axes]
            stride: Optional[int] = 1

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        locator = matplotlib.dates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = matplotlib.dates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # plot
        dates = [dt.datetime.fromtimestamp(posix, tz=dt.timezone.utc) for posix in self.state_trajectory[:-1:stride, 2]]
        ax.step(dates, self.action_trajectory[::stride, 0], where='post', label='u_power')
        ax.step(dates, self.action_trajectory[::stride, 1], where='post', label='angle')

        plt.title('Simulator Control Trajectory')
        plt.ylabel('u_power and angle in units')
        plt.xlabel('time')

        return ax

    def get_lon_lat_time_interval(self, end_region: Optional[SpatialPoint] = None,
                                  margin: Optional[float] = 0) -> Tuple:
        """
        Helper function to find the interval around start/trajectory/goal.
        Args:
            end_region: Optional[SpatialPoint]
            margin: Optional[float]

        Returns:
            lon_interval:  [x_lower, x_upper] in degrees
            lat_interval:  [y_lower, y_upper] in degrees
            time_interval: [t_lower, t_upper] in posix time
        """
        if end_region is None:
            lon_min = np.min(self.state_trajectory[:, 0])
            lon_max = np.max(self.state_trajectory[:, 0])
            lat_min = np.min(self.state_trajectory[:, 1])
            lat_max = np.max(self.state_trajectory[:, 1])
        else:
            lon_min = min([np.min(self.state_trajectory[:, 0]), end_region.lon.deg])
            lon_max = max([np.max(self.state_trajectory[:, 0]), end_region.lon.deg])
            lat_min = min([np.min(self.state_trajectory[:, 1]), end_region.lat.deg])
            lat_max = max([np.max(self.state_trajectory[:, 1]), end_region.lat.deg])

        return [lon_min - margin, lon_max + margin], [lat_min - margin, lat_max + margin], [self.state_trajectory[0, 2],
                                                                                            self.state_trajectory[
                                                                                                -1, 2]]

    def get_index_from_posix_time(self, posix_time: float) -> int:
        """
        Helper function to find the closest trajectory index corresponding to a given posix time.
        Args:
            posix_time: float

        Returns:
            index: float
        """
        lon_interval, lat_interval, time_interval = self.get_lon_lat_time_interval()

        if posix_time <= time_interval[0]:
            index = 0
        elif posix_time >= time_interval[1]:
            index = -1
        else:
            index = np.searchsorted(a=self.state_trajectory[:, 2], v=posix_time)
            # index = np.argwhere(self.state_trajectory[:, 2] == posix_time).flatten()
            # index = 0 if index.size == 0 else int(index[0])

        return index