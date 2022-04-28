"""A Ocean Platform Arena.
A Ocean arena contains the logic for navigating a platform in the ocean.
"""

import dataclasses
import datetime as dt
from typing import Dict, Optional
import matplotlib.axes
import numpy as np
from matplotlib import pyplot as plt


from ocean_navigation_simulator.env.PlatformState import SpatialPoint
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.env.data_sources.SolarIrradianceField import SolarIrradianceField
from ocean_navigation_simulator.env.data_sources.SeaweedGrowthField import SeaweedGrowthField
from ocean_navigation_simulator.env.Platform import Platform, PlatformState, PlatformAction
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector


@dataclasses.dataclass
class ArenaObservation:
    """
    Specifies an observation from the simulator.
    This differs from SimulatorState in that the observations are not
    ground truth state, and are instead noisy observations from the
    environment.
    """
    platform_state: PlatformState                       # position, time, battery
    true_current_at_state: OceanCurrentVector           # measured current at platform_state
    forecasted_current_at_state: OceanCurrentVector     # forecasted current at platform_state


class Arena:
    """A OceanPlatformArena in which an ocean platform moves through a current field."""

    # TODO: where do we do the reset? I guess for us reset mostly would mean new start and goal position?
    # TODO: not sure what that should be for us, decide where to put the feature constructor
    def __init__(
            self,
            sim_cache_dict: Dict, platform_dict: Dict,
            ocean_dict: Dict,
            solar_dict: Optional[Dict] = None,
            seaweed_dict: Optional[Dict] = None
    ):
        """OceanPlatformArena constructor.
    Args:
        sim_cache_dict:
        platform_dict:
        ocean_dict:
        solar_dict:
        seaweed_dict:
    Optional Args:
        geographic_coordinate_system: If True we use the Geographic coordinate system in lat, lon degree, if false the spatial system is in meters in x, y.
    """
        self.ocean_field = OceanCurrentField(
            sim_cache_dict=sim_cache_dict,
            hindcast_source_dict=ocean_dict['hindcast'],
            forecast_source_dict=ocean_dict['forecast'],
            use_geographic_coordinate_system=platform_dict['use_geographic_coordinate_system']
        )

        if solar_dict is not None and solar_dict['hindcast'] is not None:
            self.solar_field = SolarIrradianceField(
                sim_cache_dict=sim_cache_dict,
                hindcast_source_dict=solar_dict['hindcast'],
                forecast_source_dict=solar_dict['forecast'],
                use_geographic_coordinate_system=platform_dict['use_geographic_coordinate_system']
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
                use_geographic_coordinate_system=platform_dict['use_geographic_coordinate_system']
            )
        else:
            self.seaweed_field = None

        self.platform = Platform(
            platform_dict=platform_dict,
            ocean_source=self.ocean_field.hindcast_data_source,
            solar_source=self.solar_field.hindcast_data_source if self.solar_field is not None else None,
            seaweed_source=self.seaweed_field.hindcast_data_source if self.seaweed_field is not None else None
        )

        self.initial_state, self.state_trajectory, self.action_trajectory = [None]*3

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
        # TODO: Shall we keep those trajectories as np arrays or log them also as objects which we can transfer back
        # and forth to numpy arrays when we want to?

        self.state_trajectory = np.expand_dims(np.array(platform_state).squeeze(), axis=0)
        self.action_trajectory = np.zeros(shape=(0, 2))

        return ArenaObservation(
            platform_state=platform_state,
            true_current_at_state=self.ocean_field.get_ground_truth(self.initial_state.to_spatio_temporal_point()),
            forecasted_current_at_state=self.ocean_field.get_forecast(self.initial_state.to_spatio_temporal_point())
        )

    def step(self, action: PlatformAction) -> ArenaObservation:
        """Simulates the effects of choosing the given action in the system.
    Args:
        action: The action to take in the simulator.
    Returns:
        Arena Observation including platform state, true current at platform, forecasts
    """
        state = self.platform.simulate_step(action)

        self.state_trajectory = np.append(self.state_trajectory, np.expand_dims(np.array(state).squeeze(), axis=0), axis=0)
        self.action_trajectory = np.append(self.action_trajectory, np.expand_dims(np.array(action).squeeze(), axis=0), axis=0)

        return ArenaObservation(
            platform_state=state,
            true_current_at_state=self.ocean_field.get_ground_truth(state.to_spatio_temporal_point()),
            forecasted_current_at_state=self.ocean_field.get_forecast(state.to_spatio_temporal_point())
        )

    def quick_plot(
            self,
            end_region: Optional[SpatialPoint] = None
    ):
        import time
        start = time.time()
        self.plot_spatial(background='currents', end_region=end_region, margin=2, control_stride=100).get_figure().show()
        #self.plot_spatial(end_region=end_region, margin=2, background='solar')
        #self.plot_spatial(end_region=end_region, margin=2, background='seaweed')
        self.plot_battery().get_figure().show()
        self.plot_seaweed().get_figure().show()
        self.plot_control().get_figure().show()
        self.animate_spatial(end_region=end_region, show_control=True, control_stride=100)

        print("Create Plot: ", time.time() - start)

    def plot_spatial(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        index: Optional[int] = None,
        background: Optional[str] = 'current',
        end_region: Optional[SpatialPoint] = None,
        problem: Optional[Problem] = None,
        show_trajectory: Optional[bool] = True,
        show_control: Optional[bool] = True,
        margin: Optional[float] = 0,
        trajectory_stride: Optional[int] = 1,
        control_stride: Optional[int] = 1,
    ):
        # Intervals
        lon_interval, lat_interval = self.get_lon_lat_interval(margin=margin, end_region=end_region)

        # Background
        if background == 'current' or background == 'currents':
            ax = self.ocean_field.hindcast_data_source.plot_currents_at_time(
                time=self.state_trajectory[0, 2],
                x_interval=lon_interval,
                y_interval=lat_interval,
                plot_type='quiver',
                return_ax=True,
                max_spatial_n=120
            )
        elif background == 'solar':
            ax = self.solar_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2],
                x_interval=lon_interval,
                y_interval=lat_interval,
                plot_type='quiver',
                return_ax=True,
                max_spatial_n=120
            )
        elif background == 'seaweed' or background == 'growth':
            ax = self.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2],
                x_interval=lon_interval,
                y_interval=lat_interval,
                plot_type='quiver',
                return_ax=True,
                max_spatial_n=120
            )
        elif ax is None:
            fig, ax = plt.subplots()

        # Problem
        if problem is not None:
            ax = problem.plot(ax)

        # Current Position
        if index is not None:
            ax.scatter(self.state_trajectory[index, 0], self.state_trajectory[index, 1], c='black', marker='o', s=500, label='position')

        # Trajectory
        if show_trajectory:
            ax.plot(self.state_trajectory[::trajectory_stride, 0], self.state_trajectory[::trajectory_stride, 1], '-', marker='x', markersize=1, color='black', linewidth=2, label='trajectory')

        # Control
        if show_control:
            u_vec = self.action_trajectory[::control_stride, 0] * np.cos(self.action_trajectory[::control_stride, 1])
            v_vec = self.action_trajectory[::control_stride, 0] * np.sin(self.action_trajectory[::control_stride, 1])
            ax.quiver(self.state_trajectory[:-1:control_stride, 0], self.state_trajectory[:-1:control_stride, 1], u_vec, v_vec, color='m', scale=15)

        return ax

    def animate_spatial(
        self,
        background: Optional[str] = 'current',
        end_region: Optional[SpatialPoint] = None,
        show_trajectory: Optional[bool] = True,
        show_control: Optional[bool] = True,
        margin: Optional[float] = 0,
        trajectory_stride: Optional[int] = 1,
        control_stride: Optional[int] = 1,
    ):
        # Intervals
        lon_interval, lat_interval = self.get_lon_lat_interval(margin=margin, end_region=end_region)
        time_interval = [self.state_trajectory[0, 2], self.state_trajectory[-1, 2]]

        def add_ax_func(ax, posix_time):
            if posix_time <= time_interval[0]:
                index = 0
            elif posix_time >= time_interval[1]:
                index = -1
            else:
                index = np.argwhere(self.state_trajectory[:, 2]==posix_time).flatten()
                index = 0 if index.size == 0 else int(index[0])

            self.plot_spatial(
                ax=ax,
                index=index,
                background=None,
                end_region=end_region,
                show_trajectory=show_trajectory,
                show_control=show_control,
                margin=margin,
                trajectory_stride=trajectory_stride,
                control_stride=control_stride,
            )

        if background == 'current' or background == 'currents':
            self.ocean_field.hindcast_data_source.animate_currents(
                x_interval=lon_interval,
                y_interval=lat_interval,
                t_interval=time_interval,
                save_as_filename='full_test.gif',
                #html_render='safari',
                max_spatial_n=50,
                max_temp_n=50,
                add_ax_func=add_ax_func,
            )


    def plot_battery(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        stride: Optional[int] = 1,
    ):
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

    def plot_seaweed(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        stride: Optional[int] = 1,
    ):
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

    def plot_control(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        stride: Optional[int] = 1,
    ):
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

    def get_lon_lat_interval(
            self,
            end_region: Optional[SpatialPoint] = None,
            margin: Optional[float] = 0,
    ):
        """
        Helper function to find the interval around start/stop points.
        Args:
            end_region: SpatialPoint
            margin: float

        Returns:
            lon_interval: [x_lower, x_upper] in degrees
            lat_interval: [y_lower, y_upper] in degrees
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

        return [lon_min - margin, lon_max + margin], [lat_min - margin, lat_max + margin]






