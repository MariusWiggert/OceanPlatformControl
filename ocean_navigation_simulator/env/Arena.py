"""A Ocean Platform Arena.
A Ocean arena contains the logic for navigating a platform in the ocean.
"""

import dataclasses
import string
from datetime import datetime
from typing import Dict, Optional

import matplotlib.axes
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from ocean_navigation_simulator import Problem
from ocean_navigation_simulator.env.PlatformState import SpatialPoint

from ocean_navigation_simulator.env.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.env.data_sources.SolarIrradianceField import SolarIrradianceField
from ocean_navigation_simulator.env.data_sources.SeaweedGrowthField import SeaweedGrowthField
from ocean_navigation_simulator.env.Platform import Platform, PlatformState, PlatformAction
from ocean_navigation_simulator.utils import plotting_utils, simulation_utils
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
import ocean_navigation_simulator.env.utils.units as units


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
    def __init__(self, sim_cache_dict: Dict, platform_dict: Dict, ocean_dict: Dict,
                 solar_dict: Optional[Dict] = None, seaweed_dict: Optional[Dict] = None):
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
        # Initialize the Data Fields from the respective dictionaries
        self.ocean_field = OceanCurrentField(sim_cache_dict=sim_cache_dict,
                                             hindcast_source_dict=ocean_dict['hindcast'],
                                             forecast_source_dict=ocean_dict['forecast'],
                                             use_geographic_coordinate_system=platform_dict['use_geographic_coordinate_system'])
        if solar_dict is not None and solar_dict['hindcast'] is not None:
            self.solar_field = SolarIrradianceField(sim_cache_dict=sim_cache_dict,
                                                    hindcast_source_dict=solar_dict['hindcast'],
                                                    forecast_source_dict=solar_dict['forecast'],
                                                    use_geographic_coordinate_system=platform_dict['use_geographic_coordinate_system'])
        else:
            self.solar_field = None

        if seaweed_dict is not None and seaweed_dict['hindcast'] is not None:
            # For initializing the SeaweedGrowth Field we need to supply the respective SolarIrradianceSources
            seaweed_dict['hindcast']['source_settings']['solar_source'] = self.solar_field.hindcast_data_source
            if seaweed_dict['forecast'] is not None:
                seaweed_dict['forecast']['source_settings']['solar_source'] = self.solar_field.hindcast_data_source
            self.seaweed_field = SeaweedGrowthField(sim_cache_dict=sim_cache_dict,
                                                    hindcast_source_dict=seaweed_dict['hindcast'],
                                                    forecast_source_dict=seaweed_dict['forecast'],
                                                    use_geographic_coordinate_system=platform_dict['use_geographic_coordinate_system'])
        else:
            self.seaweed_field = None

        # Initialize the Platform Object from the dictionary
        self.platform = Platform(platform_dict=platform_dict,
                                 ocean_source=self.ocean_field.hindcast_data_source,
                                 solar_source=self.solar_field.hindcast_data_source if self.solar_field is not None else None,
                                 seaweed_source=self.seaweed_field.hindcast_data_source if self.seaweed_field is not None else None)

        # Initialize variables for holding the platform and state
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
        return ArenaObservation(platform_state=platform_state,
                                true_current_at_state=self.ocean_field.get_ground_truth(
                                    self.initial_state.to_spatio_temporal_point()),
                                forecasted_current_at_state=self.ocean_field.get_forecast(
                                    self.initial_state.to_spatio_temporal_point())
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

    def quick_plot(self, end_region: Optional[SpatialPoint] = None):
        self.plot_spatial(end_region=end_region, margin=2, background='currents').get_figure().show()
        #self.plot_spatial(end_region=end_region, margin=2, background='solar')
        #self.plot_spatial(end_region=end_region, margin=2, background='seaweed')
        #self.plot_battery()
        # self.plot_seaweed(end_region=end_region, margin=2)
        # self.plot_control(end_region=end_region, margin=2)

    def plot_spatial(
            self,
            background: Optional[str] = 'current',
            end_region: Optional[SpatialPoint] = None,
            show_trajectory: Optional[bool] = True,
            show_control: Optional[bool] = True,
            margin: Optional[float] = 0,
            stride: Optional[int] = 1,
            ax: Optional[matplotlib.axes.Axes] = None,
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
                target_max_n=120
            )
        elif background == 'solar':
            ax = self.solar_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2],
                x_interval=lon_interval,
                y_interval=lat_interval,
                plot_type='quiver',
                return_ax=True,
                target_max_n=120
            )
        elif background == 'seaweed' or background == 'growth':
            ax = self.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2],
                x_interval=lon_interval,
                y_interval=lat_interval,
                plot_type='quiver',
                return_ax=True,
                target_max_n=120
            )
        else:
            fig, ax = plt.subplots()

        # Problem
        if end_region is not None:
            ax.scatter(self.state_trajectory[0, 0], self.state_trajectory[0, 1], c='r', marker='o', s=200, label='start')
            ax.scatter(end_region.lon.deg, end_region.lat.deg, c='g', marker='x', s=200, label='end')

        # Trajectory
        if show_trajectory:
            ax.plot(self.state_trajectory[::stride, 0], self.state_trajectory[::stride, 1], '-', marker='x', c='k', linewidth=2)

        # Control
        if show_control:
            u_vec = self.action_trajectory[::stride, 0] * np.cos(self.action_trajectory[::stride, 1])
            v_vec = self.action_trajectory[::stride, 0] * np.sin(self.action_trajectory[::stride, 1])
            ax.quiver(self.state_trajectory[:-1:stride, 0], self.state_trajectory[:-1:stride, 1], u_vec, v_vec, color='m', scale=15)

        return ax


    def plot_battery(
            self,
            stride: Optional[int] = 1,
            ax: Optional[matplotlib.axes.Axes] = None,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # some stuff for flexible date axis
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        # plot
        dates = [datetime.fromtimestamp(posix, tz=datetime.timezone.utc) for posix in self.state_trajectory[::stride, 2]]
        ax.plot(dates, self.state_trajectory[::stride, 3])
        # set axis and stuff
        ax.set_title('Battery charge over time')
        ax.set_ylim(0., 1.1)
        ax.set_xlabel('time in h')
        ax.set_ylabel('Battery Charging level [0,1]')

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






