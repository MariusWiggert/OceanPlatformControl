"""
    The Ocean Arena contains the logic for navigating of the platform in the ocean, the growth of
     the seaweed as well as battery usage.
"""
import dataclasses
import datetime as dt
import logging
import os
import time
from typing import (
    AnyStr,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import matplotlib
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

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
    PlatformActionSet,
)
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    PlatformStateSet,
    SpatialPoint,
    SpatioTemporalPoint,
)
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.utils.misc import get_markers
from ocean_navigation_simulator.utils.plotting_utils import (
    animate_trajectory,
    get_lon_lat_time_interval_from_trajectory,
)
from ocean_navigation_simulator.utils.units import (
    Distance,
    format_datetime_x_axis,
    haversine_rad_from_deg,
)


@dataclasses.dataclass
class ArenaObservation:
    """
    Specifies an observation from the simulator.
    This differs from SimulatorState in that the observations are not
    ground truth state, and are instead noisy observations from the
    environment.
    """

    platform_state: Union[PlatformState, PlatformStateSet]  # position, time, battery
    true_current_at_state: OceanCurrentVector  # measured current at platform_state
    forecast_data_source: Union[
        OceanCurrentSource, OceanCurrentSourceXarray, OceanCurrentSourceAnalytical
    ]  # Data Source of the forecast

    def __len__(self):
        if type(self.platform_state) is PlatformStateSet:
            return len(self.platform_state)
        else:
            return 1

    def get_single_observation(self, id: Optional[int] = 0):
        if type(self.platform_state) is PlatformState:
            return self
        else:
            return ArenaObservation(
                self.platform_state[id],
                true_current_at_state=self.true_current_at_state,
                forecast_data_source=self.forecast_data_source,
            )

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

    ocean_field: OceanCurrentField = None
    solar_field: SolarIrradianceField = None
    seaweed_field: SeaweedGrowthField = None
    platform: Platform = None

    def __init__(
        self,
        casadi_cache_dict: Dict,
        platform_dict: Dict,
        ocean_dict: Dict,
        use_geographic_coordinate_system: bool,
        solar_dict: Optional[Dict] = None,
        seaweed_dict: Optional[Dict] = None,
        spatial_boundary: Optional[Dict] = None,
        collect_trajectory: Optional[bool] = True,
        logging_level: Optional[AnyStr] = "INFO",
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
            spatial_boundary:                dictionary containing the "x" and "y" spatial boundaries as list of [min, max]
            collect_trajectory:              boolean if True trajectory of states and actions is logged, otherwise not.
            logging_level:                   Level applied for logging.
        """
        # initialize arena logger
        self.logger = logging.getLogger("arena")
        self.logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

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
        else:
            self.solar_field = None
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
        else:
            self.seaweed_field = None

        self.logger.info(f"Arena: Generate Sources ({time.time() - start:.1f}s)")

        # Step 2: Generate Platforms simulator
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
        )

        self.logger.info(f"Arena: Generate Platform ({time.time() - start:.1f}s)")

        # Step 3: Initialize Variables
        self.spatial_boundary = spatial_boundary
        self.collect_trajectory = collect_trajectory
        self.initial_state, self.state_trajectory, self.action_trajectory = [None] * 3
        self.use_geographic_coordinate_system = use_geographic_coordinate_system

    def reset(self, platform_set: PlatformStateSet) -> ArenaObservation:
        """Resets the arena.
        Args:
            platform_state_set
        Returns:
          The first observation from the newly reset simulator
        """
        self.initial_states = platform_set
        self.platform.set_state(self.initial_states)
        self.platform.initialize_dynamics(self.initial_states)
        self.ocean_field.forecast_data_source.update_casadi_dynamics(self.initial_states)

        # self.state_trajectory = np.expand_dims(np.array(platform_set).squeeze(), axis=0)
        self.state_trajectory = np.atleast_3d(np.array(platform_set))
        # object should be a PlatformStateSet otherwise len is not the number of platforms but the number of states
        self.action_trajectory = np.zeros(shape=(len(platform_set), 2, 0))

        observation = ArenaObservation(
            platform_state=platform_set,
            true_current_at_state=self.ocean_field.get_ground_truth(
                platform_set.to_spatio_temporal_point()
            ),
            forecast_data_source=self.ocean_field.forecast_data_source,
        )
        return observation

    def step(self, action: PlatformActionSet) -> ArenaObservation:
        """Simulates the effects of choosing the given action in the system.
        Args:
            action: The action to take in the simulator.
        Returns:
            Arena Observation including platform state, true current at platform, forecasts
        """
        platform_set = self.platform.simulate_step(action)

        if self.collect_trajectory:
            # self.state_trajectory = np.append(
            #     self.state_trajectory,
            #     np.expand_dims(np.array(platform_set).squeeze(), axis=0),
            #     axis=0,
            # )
            self.state_trajectory = np.append(
                self.state_trajectory, np.atleast_3d(np.array(platform_set)), axis=2
            )
            # self.action_trajectory = np.append(
            #     self.action_trajectory, np.expand_dims(np.array(action).squeeze(), axis=0), axis=0
            # )
            self.action_trajectory = np.append(
                self.action_trajectory, np.atleast_3d(np.array(action)), axis=2
            )
        return ArenaObservation(
            platform_state=platform_set,
            true_current_at_state=self.ocean_field.get_ground_truth(
                platform_set.to_spatio_temporal_point()
            ),
            forecast_data_source=self.ocean_field.forecast_data_source,
        )

    def is_inside_arena(self, margin: Optional[float] = 0.0) -> bool:
        """Check if the current platform state is within the arena spatial boundary."""
        if self.spatial_boundary is not None:
            inside_x = (
                self.spatial_boundary["x"][0].deg + margin
                < self.platform.state_set.lon.deg
                < self.spatial_boundary["x"][1].deg - margin
            )
            inside_y = (
                self.spatial_boundary["y"][0].deg + margin
                < self.platform.state_set.lat.deg
                < self.spatial_boundary["y"][1].deg - margin
            )
            return inside_x and inside_y
        return True

    def is_on_land(self, point: SpatialPoint = None) -> bool:
        """Returns True/False if the closest grid_point to the self.cur_state is on_land."""
        if point is None:
            point = self.platform.state_set
        return self.ocean_field.hindcast_data_source.is_on_land(point)

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
        """
        if self.is_on_land():
            return -2
        elif check_inside and not self.is_inside_arena(margin):
            return -3
        else:
            return problem.is_done(self.platform.state_set)

    def problem_status_text(self, problem_status):
        """
        Get a text to the problem status.Can be used for debugging.
        Returns:
            'Success'       if problem was solved
            'Running'       if problem is still open
            'Timeout'       if problem timed out
            'Stranded'      if platform stranded
            'Outside Arena' if platform left specified araena region (spatial boundaries)
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
        else:
            return "Invalid"

    ### Various Plotting Functions for the Arena Object ###

    # general form of state trajectory array:
    # [nb_platforms, nb_states, time]
    # action_trajectory: [#nb_platforms, 2, time]
    def plot_control_trajectory_on_map(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        color: Optional[str] = "magenta",
        stride: Optional[int] = 1,
        control_vec_scale: Optional[int] = 10,
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

        for k in range(self.state_trajectory.shape[0]):
            u_vec = self.action_trajectory[k, 0, ::stride] * np.cos(
                self.action_trajectory[k, 1, ::stride]
            )
            v_vec = self.action_trajectory[k, 0, ::stride] * np.sin(
                self.action_trajectory[k, 1, ::stride]
            )
            ax.quiver(
                self.state_trajectory[k, 0, :-1:stride],
                self.state_trajectory[k, 1, :-1:stride],
                u_vec,
                v_vec,
                color=color,
                scale=control_vec_scale,
                angles="xy",
                label="Control Input of platform" if k == 0 else "",
            )

        return ax

    def animate_trajectory(
        self,
        margin: Optional[float] = 1,
        problem: Optional[NavigationProblem] = None,
        temporal_resolution: Optional[float] = None,
        add_ax_func_ext: Optional[Callable] = None,
        output: Optional[AnyStr] = "traj_animation.mp4",
        **kwargs,
    ):
        """Plotting functions to animate the trajectory of the arena so far.
        Optional Args:
              margin:            Margin as box around x_0 and x_T to plot
              problem:           Navigation Problem object
              temporal_resolution:  The temporal resolution of the animation in seconds (per default same as data_source)
              add_ax_func_ext:  function handle what to add on top of the current visualization
                                signature needs to be such that it takes an axis object and time as input
                                e.g. def add(ax, time, x=10, y=4): ax.scatter(x,y) always adds a point at (10, 4)
              # Other variables possible via kwargs see DataSource animate_data, such as:
              fps:              Frames per second
              output:           How to output the animation. Options are either saved to file or via html in jupyter/safari.
                                Strings in {'*.mp4', '*.gif', 'safari', 'jupyter'}
              forward_time:     If True, animation is forward in time, if false backwards
              **kwargs:         Further keyword arguments for plotting(see plot_currents_from_xarray)
        """
        # shallow wrapper to plotting utils function
        animate_trajectory(
            state_trajectory=self.state_trajectory,
            ctrl_trajectory=self.action_trajectory,
            data_source=self.ocean_field.hindcast_data_source,
            problem=problem,
            margin=margin,
            temporal_resolution=temporal_resolution,
            add_ax_func_ext=add_ax_func_ext,
            output=output,
            **kwargs,
        )

    def plot_state_trajectory_on_map(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        color: Optional[str] = "black",
        stride: Optional[int] = 1,
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
        for k in range(self.state_trajectory.shape[0]):
            ax.plot(
                self.state_trajectory[k, 0, ::stride],
                self.state_trajectory[k, 1, ::stride],
                "-",
                marker=".",
                markersize=1,
                color=color,
                linewidth=1,
                label="State Trajectory" if k == 0 else "",
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
        background: Optional[str] = "current",
        index: Optional[int] = -1,
        show_current_position: Optional[bool] = True,
        current_position_color: Optional[str] = "black",
        # State
        show_state_trajectory: Optional[bool] = True,
        state_color: Optional[str] = "black",
        # Control
        show_control_trajectory: Optional[bool] = True,
        control_color: Optional[str] = "magenta",
        control_stride: Optional[int] = 1,
        # Problem (Target)
        problem: Optional[Problem] = None,
        problem_start_color: Optional[str] = "red",
        problem_target_color: Optional[str] = "green",
        x_interval: Optional[List] = None,
        y_interval: Optional[List] = None,
        margin: Optional[int] = 1,
        # plot directly or return ax
        return_ax: Optional[bool] = False,
    ) -> matplotlib.axes.Axes:
        """Helper Function to plot everything together on a map."""
        if x_interval is None or y_interval is None:
            x_interval, y_interval, _ = get_lon_lat_time_interval_from_trajectory(
                state_trajectory=self.state_trajectory, margin=margin
            )

        # Background: Data Sources
        if "current" in background:
            ax = self.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2, index],  # end time for platform 0
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
            )
        elif "solar" in background:
            ax = self.solar_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2, index],
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
            )
        elif "seaweed" in background or "growth" in background:
            ax = self.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2, index],
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
            )
        else:
            raise Exception(
                f"Arena: Background '{background}' is not avaialble only 'current', 'solar' or 'seaweed."
            )

        if show_state_trajectory:
            self.plot_state_trajectory_on_map(ax=ax, color=state_color)
        if show_control_trajectory:
            self.plot_control_trajectory_on_map(ax=ax, color=control_color, stride=control_stride)
        if show_current_position:
            markers = get_markers()
            for k in range(self.state_trajectory.shape[0]):
                ax.scatter(
                    self.state_trajectory[k, 0, index],
                    self.state_trajectory[k, 1, index],
                    c=current_position_color,
                    marker=next(markers),
                    s=100,
                    label=f"current position platform {k}",
                )
        if problem is not None:
            problem.plot(ax=ax)

        ax.yaxis.grid(color="gray", linestyle="dashed")
        ax.xaxis.grid(color="gray", linestyle="dashed")
        ax.legend(loc=4, prop={"size": 4}, numpoints=1)

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
            dt.datetime.fromtimestamp(posix, tz=dt.timezone.utc)
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
            dt.datetime.fromtimestamp(posix, tz=dt.timezone.utc)
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
            dt.datetime.fromtimestamp(posix, tz=dt.timezone.utc)
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

    def plot_distance_evolution_between_neighbors(
        self, figsize: Tuple[int] = (8, 6)
    ) -> matplotlib.axes.Axes:
        """
        Function to compute distance evolution between neighbors over their trajectories
        For now just handle two platforms
        """
        idx_state = {
            "lon": 0,
            "lat": 1,
            "time": 2,
        }
        d = Distance(
            rad=haversine_rad_from_deg(
                self.state_trajectory[0, idx_state["lon"], :],
                self.state_trajectory[0, idx_state["lat"], :],
                self.state_trajectory[1, idx_state["lon"], :],
                self.state_trajectory[1, idx_state["lat"], :],
            )
        )
        plt.figure(figsize=figsize)
        ax = plt.axes()
        t0 = dt.datetime.fromtimestamp(self.state_trajectory[0, 2, 0], tz=dt.timezone.utc)
        dates = [
            dt.datetime.fromtimestamp(posix, tz=dt.timezone.utc)
            for posix in self.state_trajectory[0, 2, ::1]
        ]
        vect_strftime = np.vectorize(dt.datetime.strftime)
        dates = vect_strftime(dates, "%H:%M")
        ax.plot(dates, d.km, "--x")
        plt.ylabel("distance in km")
        plt.title("Distance evolution between platforms over time [h]")
        plt.xlabel("time")
