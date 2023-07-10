"""
    The Ocean Arena contains the logic for navigating of the platform in the ocean, the growth of
     the seaweed as well as battery usage.
"""
import dataclasses
import datetime
import logging
import time
from typing import AnyStr, Callable, Dict, List, Literal, Optional, Union

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt

from ocean_navigation_simulator.data_sources.Bathymetry.BathymetrySource import (
    BathymetrySource2d,
)
from ocean_navigation_simulator.data_sources.GarbagePatch.GarbagePatchSource import (
    GarbagePatchSource2d,
)
from ocean_navigation_simulator.data_sources.OceanCurrentField import (
    OceanCurrentField,
)
from ocean_navigation_simulator.data_sources.OceanCurrentSource.AnalyticalOceanCurrents import (
    OceanCurrentSourceAnalytical,
)
from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import (
    OceanCurrentSource,
    OceanCurrentSourceXarray,
    OceanCurrentVector,
)
from ocean_navigation_simulator.data_sources.SeaweedGrowthField import (
    SeaweedGrowthField,
)
from ocean_navigation_simulator.data_sources.SolarIrradianceField import (
    SolarIrradianceField,
)
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import (
    Platform,
    PlatformAction,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.utils.misc import timing_logger
from ocean_navigation_simulator.utils.plotting_utils import (
    animate_trajectory,
    get_lon_lat_time_interval_from_trajectory,
)
from ocean_navigation_simulator.utils.units import format_datetime_x_axis


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
        OceanCurrentSource, OceanCurrentSourceXarray, OceanCurrentSourceAnalytical
    ]  # Data Source of the forecast

    def replace_spatio_temporal_point(self, point: SpatioTemporalPoint):
        """
        this function is required to use the hindcast planner
        TODO: change HJ planner to directly accept datasources
        """
        return ArenaObservation(
            platform_state=PlatformState(
                lon=point.lon,
                lat=point.lat,
                date_time=point.date_time,
                battery_charge=self.platform_state.battery_charge,
                seaweed_mass=self.platform_state.seaweed_mass,
            ),
            true_current_at_state=self.true_current_at_state,
            forecast_data_source=self.forecast_data_source,
        )

    def replace_datasource(
        self,
        datasource: Union[
            OceanCurrentSource, OceanCurrentSourceXarray, OceanCurrentSourceAnalytical
        ],
    ):
        """
        this function is required to use the hindcast planner
        TODO: change HJ planner to directly accept datasources
        """
        return ArenaObservation(
            platform_state=self.platform_state,
            true_current_at_state=self.true_current_at_state,
            forecast_data_source=datasource,
        )


class Arena:
    """A OceanPlatformArena in which an ocean platform moves through a current field."""

    # TODO: we never need this if it is none. why do we have it?
    ocean_field: OceanCurrentField = None
    solar_field: SolarIrradianceField = None
    seaweed_field: SeaweedGrowthField = None
    bathymetry_source: BathymetrySource2d = None
    garbage_source: GarbagePatchSource2d = None
    platform: Platform = None
    timeout: Union[datetime.timedelta, int] = None

    def __init__(
        self,
        casadi_cache_dict: Dict,
        platform_dict: Dict,
        ocean_dict: Dict,
        use_geographic_coordinate_system: bool,
        solar_dict: Optional[Dict] = None,
        seaweed_dict: Optional[Dict] = None,
        bathymetry_dict: Optional[Dict] = None,
        garbage_dict: Optional[Dict] = None,
        spatial_boundary: Optional[Dict] = None,
        collect_trajectory: Optional[bool] = True,
        timeout: Union[datetime.timedelta, int] = None,
    ):
        """OceanPlatformArena constructor.
        Args:
            casadi_cache_dict:               Dictionary how much data in space and time is cached for faster simulation.
                                             The area is a square with "deg_around_x_t" and forward "time_around_x_t" in seconds.
            platform_dict:                   Dictionary with platform hardware settings. Variables are
                                             - dt_in_s for simulation step size in time (seconds)
                                             - u_max_in_mps (maximum propulsion)
                                             - drag_factor, motor_efficiency (to model Energy consumption)
                                             - solar_panel_size, solar_efficiency, battery_cap_in_wh (charging via solar)
            ocean_dict:                      Dictionary containing dicts for "hindcast" and optinally "forecast" which
                                             specify the ocean current data source. Details see OceanCurrentField.
            use_geographic_coordinate_system: If True we use the Geographic coordinate system in lat, lon degree,
                                              if false the spatial system is in meters in x, y.
        Optional Args:
            solar_dict:                      Dictionary containing dicts for "hindcast" and optinally "forecast" which
                                             specify the solar irradiance data source. Details see SolarIrradianceField.
            seaweed_dict:                    Dictionary containing dicts for "hindcast" and optinally "forecast" which
                                             specify the seaweed growth data source. Details see SeaweedGrowthField.
            bathymetry_dict:                 Directory containing source, source_settings, casadi_cache_settings and
            garbage_dict:                 Directory containing source, source_settings, casadi_cache_settings and
            spatial_boundary:                dictionary containing the "x" and "y" spatial boundaries as list of [min, max]
            collect_trajectory:              boolean if True trajectory of states and actions is logged, otherwise not.
            timeout:                         integer (in seconds) or timedelta object for max sim run (None sets no limit)
        """
        # initialize arena logger
        self.logger = logging.getLogger("arena")
        self.timeout = self.format_timeout(timeout)

        # Step 1: Initialize the DataFields from the respective Dictionaries
        start = time.time()
        # Step 1.1 Ocean Field
        self.ocean_field = OceanCurrentField(
            casadi_cache_dict=casadi_cache_dict,
            hindcast_source_dict=ocean_dict["hindcast"],
            forecast_source_dict=ocean_dict["forecast"],
            use_geographic_coordinate_system=use_geographic_coordinate_system,
        )
        # Step 1.2 Solar Irradiance Field
        if solar_dict is not None and solar_dict["hindcast"] is not None:
            self.solar_field = SolarIrradianceField(
                casadi_cache_dict=casadi_cache_dict,
                hindcast_source_dict=solar_dict["hindcast"],
                forecast_source_dict=solar_dict["forecast"] if "forecast" in solar_dict else None,
                use_geographic_coordinate_system=use_geographic_coordinate_system,
            )
        # Step 1.3 Seaweed Growth Field
        if seaweed_dict is not None and seaweed_dict["hindcast"] is not None:
            # For initializing the SeaweedGrowth Field we need to supply the respective SolarIrradianceSources
            seaweed_dict["hindcast"]["source_settings"][
                "solar_source"
            ] = self.solar_field.hindcast_data_source
            if seaweed_dict["forecast"] is not None:
                seaweed_dict["forecast"]["source_settings"][
                    "solar_source"
                ] = self.solar_field.hindcast_data_source
            self.seaweed_field = SeaweedGrowthField(
                casadi_cache_dict=casadi_cache_dict,
                hindcast_source_dict=seaweed_dict["hindcast"],
                forecast_source_dict=seaweed_dict["forecast"],
                use_geographic_coordinate_system=use_geographic_coordinate_system,
            )
        # Step 1.4 Bathymetry Field
        if bathymetry_dict is not None:
            self.bathymetry_source = BathymetrySource2d(source_dict=bathymetry_dict)
        if garbage_dict is not None:
            self.garbage_source = GarbagePatchSource2d(source_dict=garbage_dict)

        self.logger.info(f"Arena: Generate Sources ({time.time() - start:.1f}s)")

        # Step 2: Generate Platform
        start = time.time()
        self.platform = Platform(
            platform_dict=platform_dict,
            ocean_source=self.ocean_field.hindcast_data_source,
            use_geographic_coordinate_system=use_geographic_coordinate_system,
            solar_source=self.solar_field.hindcast_data_source
            if self.solar_field is not None
            else None,
            seaweed_source=self.seaweed_field.hindcast_data_source
            if self.seaweed_field is not None
            else None,
            bathymetry_source=self.bathymetry_source,
            garbage_source=self.garbage_source,
        )

        self.logger.info(f"Arena: Generate Platform ({time.time() - start:.1f}s)")

        # Step 3: Initialize Variables
        self.spatial_boundary = spatial_boundary
        self.collect_trajectory = collect_trajectory
        self.initial_state, self.state_trajectory, self.action_trajectory = [None] * 3
        self.use_geographic_coordinate_system = use_geographic_coordinate_system

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

        return ArenaObservation(
            platform_state=platform_state,
            true_current_at_state=self.ocean_field.get_ground_truth(
                self.platform.state.to_spatio_temporal_point()
            ),
            forecast_data_source=self.ocean_field.forecast_data_source,
        )

    def step(self, action: PlatformAction) -> ArenaObservation:
        """Simulates the effects of choosing the given action in the system.
        Args:
            action: The action to take in the simulator.
        Returns:
            Arena Observation including platform state, true current at platform, forecasts
        """
        # TODO: add garbage patch accumulation
        with timing_logger("Platform Step ({})", self.logger, logging.DEBUG):
            state = self.platform.simulate_step(action)

        if self.collect_trajectory:
            self.state_trajectory = np.append(
                self.state_trajectory, np.expand_dims(np.array(state).squeeze(), axis=0), axis=0
            )
            self.action_trajectory = np.append(
                self.action_trajectory, np.expand_dims(np.array(action).squeeze(), axis=0), axis=0
            )

        with timing_logger("Create Observation ({})", self.logger, logging.DEBUG):
            obs = ArenaObservation(
                platform_state=state,
                true_current_at_state=self.ocean_field.get_ground_truth(
                    state.to_spatio_temporal_point()
                ),
                forecast_data_source=self.ocean_field.forecast_data_source,
            )
        return obs

    def is_inside_arena(self, margin: Optional[float] = 0.0) -> bool:
        """Check if the current platform state is within the arena spatial boundary."""
        if self.spatial_boundary is None:
            try:
                x_boundary = [
                    self.ocean_field.hindcast_data_source.grid_dict["x_grid"][0],
                    self.ocean_field.hindcast_data_source.grid_dict["x_grid"][-1],
                ]
                y_boundary = [
                    self.ocean_field.hindcast_data_source.grid_dict["y_grid"][0],
                    self.ocean_field.hindcast_data_source.grid_dict["y_grid"][-1],
                ]
            except BaseException:
                self.logger.warning(
                    "Arena: Hindcast Ocean Source does not have x, y grid. Not checking if inside."
                )
                return True
        else:
            x_boundary = [x.deg for x in self.spatial_boundary["x"]]
            y_boundary = [y.deg for y in self.spatial_boundary["y"]]

        # calculate if inside or outside
        inside_x = x_boundary[0] + margin < self.platform.state.lon.deg < x_boundary[1] - margin
        inside_y = y_boundary[0] + margin < self.platform.state.lat.deg < y_boundary[1] - margin
        return inside_x and inside_y

    def is_on_land(self, point: SpatialPoint = None, elevation: float = 0) -> bool:
        """Returns True/False if the closest grid_point to the self.cur_state is on_land."""
        # TODO: would need to change everywhere to take argmin to not have sampling problems due to interpolation.
        if hasattr(self.bathymetry_source, "DistanceArray"):
            point = self.platform.state
            # distance = self.bathymetry_source.DistanceArray.interp(
            #     lat=point.lat.deg, lon=point.lon.deg
            # ).data
            x_idx = (np.abs(self.bathymetry_source.DistanceArray["lon"] - point.lon.deg)).argmin()
            y_idx = (np.abs(self.bathymetry_source.DistanceArray["lat"] - point.lat.deg)).argmin()
            distance = self.bathymetry_source.DistanceArray[y_idx, x_idx].data
            return distance < self.bathymetry_source.source_dict["distance"]["safe_distance"]

        elif self.bathymetry_source:
            if point is None:
                point = self.platform.state.to_spatial_point()
            return self.bathymetry_source.is_higher_than(point, elevation)
        else:
            # Check if x_grid exists (not for all data sources)
            if self.ocean_field.hindcast_data_source.grid_dict.get("x_grid", None) is not None:
                if point is None:
                    point = self.platform.state
                return self.ocean_field.hindcast_data_source.is_on_land(point)
            else:
                return False

    def is_in_garbage_patch(self, point: SpatioTemporalPoint = None) -> bool:
        if self.garbage_source:
            if point is None:
                point = self.platform.state.to_spatial_point()
            return self.garbage_source.is_in_garbage_patch(point)
        else:
            return 0

    def is_timeout(self) -> bool:
        # calculate passed_seconds
        if self.timeout is not None:
            total_seconds = (
                self.platform.state.date_time - self.initial_state.date_time
            ).total_seconds()
            return total_seconds >= self.timeout.total_seconds()
        else:
            return False

    def final_distance_to_target(self, problem: NavigationProblem) -> float:
        # Step 1: calculate min distance
        total_distance = problem.distance(
            PlatformState.from_numpy(self.state_trajectory[-1, :])
        ).deg
        min_distance_to_target = total_distance - problem.target_radius
        # Step 2: Set 0 when inside and the distance when outside
        if min_distance_to_target <= 0:
            min_distance_to_target = 0.0
        return min_distance_to_target

    @staticmethod
    def format_timeout(timeout) -> Union[datetime.timedelta, None]:
        """Helper function because we want timeout to be able to be from a dict/string."""
        if isinstance(timeout, datetime.timedelta):
            return timeout
        elif timeout is not None:
            return datetime.timedelta(seconds=timeout)
        else:
            return None

    def problem_status(
        self, problem: Problem, check_inside: Optional[bool] = True, margin: Optional[float] = 0.0
    ) -> int:
        """
        Get the problem status
        Returns:
            1   if problem was solved
            0   if problem is still open
            -1  if problem timed out
            -2  if platform stranded
            -3  if platform left specified arena region (spatial boundaries)
            -4  if platform is in Garbage patch
        """
        if self.is_timeout():
            return -1
        if check_inside and not self.is_inside_arena(margin):
            return -3
        if self.is_on_land():
            return -2
        if self.is_in_garbage_patch():
            return -4
        else:
            return problem.is_done(self.platform.state)

    def problem_status_text(self, problem_status):
        """
        Get a text to the problem status.Can be used for debugging.
        Returns:
            'Success'       if problem was solved
            'Running'       if problem is still open
            'Timeout'       if problem timed out
            'Stranded'      if platform stranded
            'Outside Arena' if platform left specified araena region (spatial boundaries)
            'In garbage patch' if platform in garbage patch
            'Invalid'       otherwise
        """
        if problem_status == 1:
            return "Success"
        elif problem_status == 0:
            return "Running"
        elif problem_status == -1:
            return "Timeout"
        elif problem_status == -2:
            return "Stranded"
        elif problem_status == -3:
            return "Outside Arena"
        elif problem_status == -4:
            return "In garbage patch"
        else:
            return "Invalid"

    ### Various Plotting Functions for the Arena Object ###

    def plot_control_trajectory_on_map(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        color: Optional[str] = "magenta",
        stride: Optional[int] = 1,
        control_vec_scale: Optional[int] = 15,
    ) -> matplotlib.axes.Axes:
        """
        Plots the control trajectory (as arrows) on a spatial map. Passing in an axis is optional.
         Otherwise a new figure is created.
        Args:
            ax: Optional[matplotlib.axes.Axes]
            color: Optional[str] = 'black'
            stride: Optional[int] = 1
            control_vec_scale: Optional[int] = 15

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        u_vec = self.action_trajectory[::stride, 0] * np.cos(self.action_trajectory[::stride, 1])
        v_vec = self.action_trajectory[::stride, 0] * np.sin(self.action_trajectory[::stride, 1])
        ax.quiver(
            self.state_trajectory[:-1:stride, 0],
            self.state_trajectory[:-1:stride, 1],
            u_vec,
            v_vec,
            color=color,
            scale=control_vec_scale,
            angles="xy",
            label="Control Inputs",
            # add the order to make sure it's always plotted on top
            zorder=10,
        )

        return ax

    def animate_trajectory(
        self,
        margin: Optional[float] = 1,
        x_interval: Optional[List[float]] = None,
        y_interval: Optional[List[float]] = None,
        problem: Optional[NavigationProblem] = None,
        temporal_resolution: Optional[float] = None,
        add_ax_func_ext: Optional[Callable] = None,
        full_traj: Optional[bool] = True,
        output: Optional[AnyStr] = "traj_animation.mp4",
        **kwargs,
    ):
        """Plotting functions to animate the trajectory of the arena so far.
        Optional Args:
              margin:            Margin as box around x_0 and x_T to plot
              x_interval:        If both x and y interval are present the margin is ignored.
              y_interval:        If both x and y interval are present the margin is ignored.
              problem:           Navigation Problem object
              temporal_resolution:  The temporal resolution of the animation in seconds (per default same as data_source)
              add_ax_func_ext:  function handle what to add on top of the current visualization
                                signature needs to be such that it takes an axis object and time as input
                                e.g. def add(ax, time, x=10, y=4): ax.scatter(x,y) always adds a point at (10, 4)
              full_traj:        Boolean, True per default to disply full trajectory at all times, when False iteratively.
              # Other variables possible via kwargs see DataSource animate_data, such as:
              fps:              Frames per second
              output:           How to output the animation. Options are either saved to file or via html in jupyter/safari.
                                Strings in {'*.mp4', '*.gif', 'safari', 'jupyter'}
              forward_time:     If True, animation is forward in time, if false backwards
              **kwargs:         Further keyword arguments for plotting(see plot_currents_from_xarray)
        """
        # shallow wrapper to plotting utils function
        animate_trajectory(
            state_trajectory=self.state_trajectory.T,
            ctrl_trajectory=self.action_trajectory.T,
            data_source=self.ocean_field.hindcast_data_source,
            problem=problem,
            margin=margin,
            x_interval=x_interval,
            y_interval=y_interval,
            temporal_resolution=temporal_resolution,
            add_ax_func_ext=add_ax_func_ext,
            full_traj=full_traj,
            output=output,
            **kwargs,
        )

    def plot_state_trajectory_on_map(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        color: Optional[str] = "black",
        stride: Optional[int] = 1,
        traj_linewidth: Optional[int] = 1,
    ) -> matplotlib.axes.Axes:
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

        ax.plot(
            self.state_trajectory[::stride, 0],
            self.state_trajectory[::stride, 1],
            "-",
            marker=".",
            markersize=1,
            color=color,
            linewidth=traj_linewidth,
            label="State Trajectory",
        )

        return ax

    def plot_arena_frame_on_map(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """Helper Function to plot the arena area on the map."""
        ax.add_patch(
            patches.Rectangle(
                (self.spatial_boundary["x"][0].deg, self.spatial_boundary["y"][0].deg),
                (self.spatial_boundary["x"][1].deg - self.spatial_boundary["x"][0].deg),
                (self.spatial_boundary["y"][1].deg - self.spatial_boundary["y"][0].deg),
                linewidth=2,
                edgecolor="r",
                facecolor="none",
                label="arena frame",
            )
        )
        return ax

    def plot_all_on_map(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        background: Optional[str] = "current",
        index: Optional[int] = 0,
        show_current_position: Optional[bool] = False,
        current_position_color: Optional[str] = "black",
        # State
        show_state_trajectory: Optional[bool] = True,
        state_color: Optional[str] = "black",
        traj_linewidth: Optional[int] = 1,
        # Control
        show_control_trajectory: Optional[bool] = True,
        control_color: Optional[str] = "magenta",
        control_stride: Optional[int] = 1,
        control_vec_scale: Optional[int] = 4,
        # Problem (Target)
        problem: Optional[Problem] = None,
        problem_start_color: Optional[str] = "red",
        problem_target_color: Optional[str] = "green",
        x_interval: Optional[List] = None,
        y_interval: Optional[List] = None,
        margin: Optional[int] = 1,
        spatial_resolution: Optional[float] = 0.1,
        vmax: Optional[float] = None,
        vmin: Optional[float] = None,
        # plot directly or return ax
        return_ax: Optional[bool] = False,
        **kwargs
    ) -> matplotlib.axes.Axes:
        """Helper Function to plot everything together on a map."""
        if x_interval is None or y_interval is None:
            x_interval, y_interval, _ = get_lon_lat_time_interval_from_trajectory(
                state_trajectory=self.state_trajectory.T, margin=margin
            )

        # Background: Data Sources
        if "current" in background:
            ax = self.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[index, 2],
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
                spatial_resolution=spatial_resolution,
                vmax=vmax,vmin=vmin,
                **kwargs
            )
        elif "solar" in background:
            ax = self.solar_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[index, 2],
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
            )
        elif "seaweed" in background or "growth" in background:
            ax = self.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[index, 2],
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
            )
        elif "bathymetry" in background:
            ax = self.bathymetry_source.plot_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
            )
        elif "garbage" in background:
            ax = self.garbage_source.plot_data_over_area(
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
            )
        else:
            raise Exception(
                f"Arena: Background '{background}' is not available only 'current', 'solar', 'seaweed', 'bathymetry', or 'garbage'."
            )

        if show_state_trajectory:
            self.plot_state_trajectory_on_map(ax=ax, color=state_color ,stride=1, traj_linewidth=traj_linewidth)
        if show_control_trajectory:
            self.plot_control_trajectory_on_map(ax=ax, color=control_color, stride=control_stride,
                                                control_vec_scale=control_vec_scale)
        if show_current_position:
            ax.scatter(
                self.state_trajectory[index, 0],
                self.state_trajectory[index, 1],
                c=current_position_color,
                marker=".",
                s=100,
                label="position at time_index {} of background ".format(index),
            )
        if problem is not None:
            problem.plot(ax=ax)

        ax.yaxis.grid(color="gray", linestyle="dashed")
        ax.xaxis.grid(color="gray", linestyle="dashed")
        ax.legend(loc="lower right")

        if return_ax:
            return ax
        else:
            plt.show()

    def plot_battery_trajectory_on_timeaxis(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        stride: Optional[int] = 1,
    ) -> matplotlib.axes.Axes:
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

        format_datetime_x_axis(ax)

        dates = [
            datetime.datetime.fromtimestamp(posix, tz=datetime.timezone.utc)
            for posix in self.state_trajectory[::stride, 2]
        ]
        ax.plot(dates, self.state_trajectory[::stride, 3])

        ax.set_title("Battery charge over time")
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel("time in h")
        ax.set_ylabel("Battery Charging level [0,1]")

        return ax

    def plot_seaweed_trajectory_on_timeaxis(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        stride: Optional[int] = 1,
    ) -> matplotlib.axes.Axes:
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
        format_datetime_x_axis(ax)

        dates = [
            datetime.datetime.fromtimestamp(posix, tz=datetime.timezone.utc)
            for posix in self.state_trajectory[::stride, 2]
        ]
        ax.plot(dates, self.state_trajectory[::stride, 3], marker=".")

        ax.set_title("Seaweed Mass over Time")
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel("time in h")
        ax.set_ylabel("Seaweed Mass in kg")

        return ax

    def plot_control_trajectory_on_timeaxis(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        stride: Optional[int] = 1,
        to_plot: Optional[Literal["both", "thrust", "direction"]] = "both",
    ) -> matplotlib.axes.Axes:
        """
        Plots the control thrust/angle on a time axis. Passing in an axis is optional.
         Otherwise a new figure is created.
        Args:
            ax: Optional[matplotlib.axes.Axes]
            stride: Optional[int] = 1
            to_plot: what aspect of the control to plot ["both", "thrust", "direction"]

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        format_datetime_x_axis(ax)

        # plot
        dates = [
            datetime.datetime.fromtimestamp(posix, tz=datetime.timezone.utc)
            for posix in self.state_trajectory[:-1:stride, 2]
        ]
        if to_plot == "both" or to_plot == "thrust":
            ax.step(dates, self.action_trajectory[::stride, 0], where="post", label="u_power")
        if to_plot == "both" or to_plot == "direction":
            ax.step(dates, self.action_trajectory[::stride, 1], where="post", label="angle")

        plt.ylabel("u_power and angle in units")
        plt.title("Simulator Control Trajectory")
        plt.xlabel("time")

        return ax

    def plot_garbage_trajectory_on_timeaxis(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        stride: Optional[int] = 1,
        # TODO: expand to safety to include bathymetry violations (then bathymetry state needs to be added to trajectory)
        # to_plot: Optional[Literal["both", "garbage", "bathymetry"]] = "both",
    ) -> matplotlib.axes.Axes:
        """
        Plots the garbgae trajectory on a time axis. Passing in an axis is optional.
         Otherwise a new figure is created.
        Args:
            ax: Optional[matplotlib.axes.Axes]
            stride: Optional[int] = 1

        Returns:
            ax:  matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        format_datetime_x_axis(ax)

        # plot
        dates = [
            datetime.datetime.fromtimestamp(posix, tz=datetime.timezone.utc)
            for posix in self.state_trajectory[::stride, 2]
        ]
        ax.plot(dates, self.state_trajectory[::stride, 5])

        plt.ylabel("is_in_garbage")
        plt.title("Inside Garbage Patch Trajectory")
        plt.xlabel("time")

        return ax

    # TODO: Cache the added stuff? All images in animation will have the same overlay.
    def add_ax_func_ext_overlay(
        self,
        ax: matplotlib.axes.Axes,
        posix_time: datetime.datetime,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        x_interval = kwargs.get("x_interval", [-180, 180])
        y_interval = kwargs.get("y_interval", [-90, 90])
        masking_val_bathymetry = kwargs.get("masking_val_bathymetry", -150)

        if self.bathymetry_source:
            ax = self.bathymetry_source.plot_mask_from_xarray(
                self.bathymetry_source.get_data_over_area(
                    x_interval=x_interval, y_interval=y_interval
                ),
                ax=ax,
                masking_val=masking_val_bathymetry,
            )
        if self.garbage_source:
            ax = self.garbage_source.plot_mask_from_xarray(
                self.garbage_source.get_data_over_area(
                    x_interval=x_interval, y_interval=y_interval
                ),
                ax=ax,
            )
        return ax

    def get_datetime_from_state_trajectory(self, state_trajectory: np.ndarray):
        """
        Function returning the list of dates for a given state trajectory.
        """
        return [
            datetime.datetime.fromtimestamp(posix, tz=datetime.timezone.utc)
            for posix in state_trajectory[:, 2]
        ]

    def get_date_string_from_state_trajectory(self, state_trajectory: np.ndarray):
        """
        Function returning the list of dates for a given state trajectory as strings.
        """
        return [
            datetime.datetime.fromtimestamp(posix, tz=datetime.timezone.utc).strftime(
                "%y-%m-%d %H:%M"
            )
            for posix in state_trajectory[:, 2]
        ]

    def get_plot_data_for_wandb(self):
        dict_for_plot = {
            "timesteps": np.arange(self.state_trajectory.shape[0]),
            "dates_timestamp": self.get_datetime_from_state_trajectory(self.state_trajectory),
            "dates_string": self.get_date_string_from_state_trajectory(self.state_trajectory),
            "lon": self.state_trajectory[:, 0],
            "lat": self.state_trajectory[:, 1],
            "battery_charge": self.state_trajectory[:, 3],
            "seaweed_mass": self.state_trajectory[:, 4],
            "inside_garbage": self.state_trajectory[:, 5],
        }
        return pd.DataFrame.from_dict(dict_for_plot, orient="columns")
