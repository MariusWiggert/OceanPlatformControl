"""
    The Ocean Arena contains the logic for navigating of the platform in the ocean, the growth of
     the seaweed as well as battery usage.
"""
import dataclasses
import datetime
import logging
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
import networkx as nx
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
from ocean_navigation_simulator.environment.MultiAgent import (
    GraphObservation,
    MultiAgent,
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

from ocean_navigation_simulator.utils.misc import (
    get_markers,
    timing_logger,)
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
    graph_obs: Union[GraphObservation, None]

    def __len__(self):
        if type(self.platform_state) is PlatformStateSet:
            return len(self.platform_state)
        else:
            return 1

    def get_single_observation(self, item: Optional[int] = 0):
        if type(self.platform_state) is PlatformState:
            return self
        else:
            return ArenaObservation(
                self.platform_state[item],
                true_current_at_state=self.true_current_at_state[item],
                forecast_data_source=self.forecast_data_source,
                graph_obs=self.graph_obs,
            )

    def __getitem__(self, item):
        return self.get_single_observation(item)

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
    timeout: Union[datetime.timedelta, int] = None

    def __init__(
        self,
        casadi_cache_dict: Dict,
        platform_dict: Dict,
        ocean_dict: Dict,
        use_geographic_coordinate_system: bool,
        solar_dict: Optional[Dict] = None,
        seaweed_dict: Optional[Dict] = None,
        spatial_boundary: Optional[Dict] = None,
        network_graph_dict: Optional[Dict] = None,
        multi_agent_graph_edges: Optional[List] = None,
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

        # Step 3: Initialize graph network for multi-agent
        if network_graph_dict is not None:
            self.multi_agent_net = MultiAgent(
                network_graph_dict=network_graph_dict,
                graph_edges=multi_agent_graph_edges
                if multi_agent_graph_edges is not None
                else None,
            )
            self.is_multi_agent = True

        # Step 4: Initialize Variables
        self.spatial_boundary = spatial_boundary
        self.collect_trajectory = collect_trajectory
        self.initial_state, self.state_trajectory, self.action_trajectory = [None] * 3
        self.use_geographic_coordinate_system = use_geographic_coordinate_system
        self.multi_agent_G_list = [None]
        self.graph_edges, self.from_nodes, self.to_nodes = None, None, None
        self.dt_in_s = platform_dict["dt_in_s"]

    def reset(self, platform_set: PlatformStateSet) -> ArenaObservation:
        """Resets the arena.
        Args:
            platform_state_set
        Returns:
          The first observation from the newly reset simulator
        """
        self.platform.initialize_dynamics(platform_set)
        self.ocean_field.forecast_data_source.update_casadi_dynamics(platform_set)
        self.initial_states = self.platform.set_state(platform_set)

        self.state_trajectory = np.atleast_3d(np.array(platform_set))
        # object should be a PlatformStateSet otherwise len is not the number of platforms but the number of states
        self.action_trajectory = np.zeros(shape=(len(platform_set), 2, 0))
        if self.is_multi_agent:
            self.multi_agent_G_list[0] = self.multi_agent_net.set_graph(platform_set=platform_set)

        observation = ArenaObservation(
            platform_state=platform_set,
            true_current_at_state=self.ocean_field.get_ground_truth(
                platform_set.to_spatio_temporal_point()
            ),
            forecast_data_source=self.ocean_field.forecast_data_source,
            graph_obs=self.multi_agent_G_list[0] if self.is_multi_agent else None,
        )
        return observation

    def step(self, action: PlatformActionSet) -> ArenaObservation:
        """Simulates the effects of choosing the given action in the system.
        Args:
            action: The action to take in the simulator.
        Returns:
            Arena Observation including platform state, true current at platform, forecasts
        """

        with timing_logger("Platform Step ({})", self.logger, logging.DEBUG):
            platform_set = self.platform.simulate_step(action)

        if self.collect_trajectory:
            self.state_trajectory = np.append(
                self.state_trajectory, np.atleast_3d(np.array(platform_set)), axis=2
            )
            self.action_trajectory = np.append(
                self.action_trajectory, np.atleast_3d(np.array(action)), axis=2
            )
        if self.is_multi_agent:
            graph_observation = self.multi_agent_net.update_graph(platform_set=platform_set)
            self.multi_agent_G_list.append(graph_observation)

        with timing_logger("Create Observation ({})", self.logger, logging.DEBUG):
            obs =  ArenaObservation(
                platform_state=platform_set,
                true_current_at_state=self.ocean_field.get_ground_truth(
                    platform_set.to_spatio_temporal_point()
                ),
                forecast_data_source=self.ocean_field.forecast_data_source,
                graph_obs=graph_observation,
            )
        return obs

    def is_inside_arena(self, platform_id: int = 0, margin: Optional[float] = 0.0) -> bool:
        # TODO: Check again if works for MultiAgent
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
        inside_x = x_boundary[0] + margin < self.platform.state_set[platform_id].lon.deg < x_boundary[1] - margin
        inside_y = y_boundary[0] + margin < self.platform.state_set[platform_id].lat.deg < y_boundary[1] - margin
        return inside_x and inside_y

    def is_on_land(self, point: SpatialPoint = None, platform_id: int = 0) -> bool:
        """Returns True/False if the closest grid_point to the self.cur_state is on_land."""
        # Check if x_grid exists (not for all data sources)
        if self.ocean_field.hindcast_data_source.grid_dict.get("x_grid", None) is not None:
            if point is None:
                point = self.platform.state_set[platform_id]
            return self.ocean_field.hindcast_data_source.is_on_land(point)
        else:
            return False

    def is_timeout(self, platform_id: int = 0) -> bool:
        # calculate passed_seconds
        if self.timeout is not None:
            total_seconds = (
                self.platform.state_set[platform_id].date_time - self.initial_state.date_time
            ).total_seconds()
            return total_seconds >= self.timeout.total_seconds()
        else:
            return False

    def final_distance_to_target(self, problem: NavigationProblem) -> float:
        # TODO: adapt for multi-agent
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
        self,
        problem: Problem,
        platform_id: int = 0,
        check_inside: Optional[bool] = True,
        margin: Optional[float] = 0.0,
    ) -> int:
        """
        Get the problem status
        # for multi agent return a problem status for platform individually
        Returns:
            1   if problem was solved
            0   if problem is still open
            -1  if problem timed out
            -2  if platform stranded
            -3  if platform left specified arena region (spatial boundaries)
        """
        if self.is_timeout():
            return -1
        if self.is_on_land(platform_id=platform_id):
            return -2
        elif check_inside and not self.is_inside_arena(platform_id=platform_id, margin=margin):
            return -3
        else:
            return problem.is_done(self.platform.state_set[platform_id])

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

    def get_stride_for_xaxis_from_temp_res(self, temporal_res: int, xticks_temporal_res: int):
        """
        Function to get the stride to parse trajectories array given a certain temporal resolution
        temporal_res: parse trajectories array to plot at fixed time samples defined by this variable
        xticks_temporal_res: temporal resolution can be different to label xticks (generally smaller than temporal_res to avoid overlapping of dates)
        """
        stride_temporal_res = int(temporal_res / self.dt_in_s)
        stride_xticks = int(xticks_temporal_res / self.dt_in_s)
        return stride_temporal_res, stride_xticks

    def get_datetime_from_state_trajectory(self, state_trajectory: np.ndarray):
        """
        Function returning the list of dates for a given state trajectory, times at which the states of a set of platform was recorded
        For now: assume all platforms are sampled at the same time, so it is sufficient to obtain the datatime of the first platform with index 0
        """
        return [
            datetime.datetime.fromtimestamp(posix, tz=datetime.timezone.utc)
            for posix in state_trajectory[0, 2, ::1]
        ]

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
            state_trajectory=self.state_trajectory,
            ctrl_trajectory=self.action_trajectory,
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
        ax: Optional[matplotlib.axes.Axes] = None,
        background: Optional[str] = "current",
        index: Optional[int] = 0,
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
        figsize: Optional[Tuple[int]] = (8, 6),
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
                figsize=figsize,
            )
        elif "solar" in background:
            ax = self.solar_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2, index],
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
                figsize=figsize,
            )
        elif "seaweed" in background or "growth" in background:
            ax = self.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
                time=self.state_trajectory[0, 2, index],
                x_interval=x_interval,
                y_interval=y_interval,
                return_ax=True,
                figsize=figsize,
            )
        else:
            raise Exception(
                f"Arena: Background '{background}' is not available only 'current', 'solar' or 'seaweed."
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
        # TODO: Adapt to MULTIAGENT
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
        # TODO: Adapt to MULTIAGENT
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
        # TODO: Adapt to MULTIAGENT
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

    def plot_distance_evolution_between_platforms(
        self,
        only_connected_comp: Optional[bool] = False,
        neighbors_list_to_plot: Optional[List[Tuple]] = None,
        figsize: Optional[Tuple[int]] = (12, 8),
        temporal_res: Optional[int] = 1800,
        xticks_temporal_res: Optional[int] = 14400,
        plot_threshold: Optional[bool] = True,
    ) -> matplotlib.axes.Axes:
        """Function to compute distance evolution between neighboring platforms over their trajectories

        Args:
            only_connected_comp (Optional[ bool ], optional): True if only the platform pairs with distance smaller than the
                                                            communication threshold are plotted (the connected pairs).
                                                            Defaults to False.
            neighbors_list_to_plot (Optional[List[Tuple]], optional): If only distances for specific pairs of platforms
                                                                    want to be plotted. Defaults to None.
            figsize (Optional[Tuple[int]], optional): Defaults to (12, 8).
            temporal_res (Optional[int], optional): sampling interval for plotting from the trajectories in [s].
                                                    Defaults to 1800 (=30mins).
            xticks_temporal_res (Optional[int], optional): xticks interval in [s]: Defaults to 14400 (= 4H).
            plot_threshold (Optional[bool], optional):  Defaults to True to show the communication and
                                                        collisions thresholds.

        Returns:
            matplotlib.axes.Axes
        """
        stride_temporal_res, stride_xticks = self.get_stride_for_xaxis_from_temp_res(
            temporal_res=temporal_res, xticks_temporal_res=xticks_temporal_res
        )
        dates = self.get_datetime_from_state_trajectory(state_trajectory=self.state_trajectory)

        ax = self.multi_agent_net.plot_distance_evolution_between_neighbors(
            list_of_graph=[G.G_communication for G in self.multi_agent_G_list]
            if only_connected_comp
            else [G.G_complete for G in self.multi_agent_G_list],
            dates=dates,
            neighbors_list_to_plot=neighbors_list_to_plot,
            stride_temporal_res=stride_temporal_res,
            stride_xticks=stride_xticks,
            figsize=figsize,
            plot_threshold=plot_threshold,
        )
        return ax

    def plot_graph_for_platform_state(
        self,
        G: nx.Graph,
        platform_set: PlatformStateSet,
        collision_communication_thrsld: Optional[Tuple] = None,
        plot_ax_ticks: Optional[bool] = False,
        figsize: Optional[Tuple] = (10, 10),
        normalize_positions: Optional[bool] = False,
        margin: Optional[int] = 1,
    ) -> matplotlib.axes.Axes:
        """_summary_

        Args:
            G (nx.Graph): Graph network to plot
            platform_set (PlatformStateSet): the set of platforms corresponding to this graph
            collision_communication_thrsld (Optional[Tuple], optional): Specific thresholds to represent
                                                                        edges as connected or risks of collisions.
                                                                        Defaults to None, later specified by the multi_agent_net instances
            plot_ax_ticks (Optional[bool], optional): Plot x and y ticks (lon, lat). Defaults to False.
            figsize (Optional[Tuple], optional): _description_. Defaults to (10, 10).
            normalize_positions (Optional[bool], optional): To normalize positions between 0 and 1. Defaults to False.
            margin (Optional[int], optional): margin from the most (left, right, bottom, top) node to the edges.
                                            Defaults to 1.

        Returns:
            matplotlib.axes.Axes
        """

        pos = {}
        x_interval, y_interval, t_interval = get_lon_lat_time_interval_from_trajectory(
            state_trajectory=np.atleast_3d(np.array(platform_set)), margin=margin
        )
        keys = platform_set.get_nodes_list()

        for lon, lat, key in zip(platform_set.lon.deg, platform_set.lat.deg, keys):
            if normalize_positions:  # if we want to have normalized node positions between 0 and 1
                pos[key] = (
                    (lon - x_interval[0]) / (x_interval[1] - x_interval[0]),
                    (lat - y_interval[0]) / (y_interval[1] - y_interval[0]),
                )  # normalize positions
            else:
                pos[key] = (lon, lat)

        t = datetime.datetime.fromtimestamp(t_interval[0], tz=datetime.timezone.utc)
        plt.figure(figsize=figsize)
        ax = self.multi_agent_net.plot_network_graph(
            G,
            pos=pos,
            t_datetime=t,
            collision_communication_thrslds=collision_communication_thrsld,
            plot_ax_ticks=plot_ax_ticks,
        )
        return ax

    def animate_graph_net_trajectory(
        self,
        collision_communication_thrslds: Optional[Tuple] = None,
        temporal_resolution: Optional[float] = None,
        plot_ax_ticks: Optional[bool] = False,
        output: Optional[AnyStr] = "network_graph_anim.mp4",
        **kwargs,
    ):
        """Animate the platforms interactions as a graph, through simulation:
            visualize edge connections, weights and collision risks

        Args:
            collision_communication_thrslds (Optional[Tuple], optional): To give different collision communication
                                                                         thresholds as in the multi_agent_net instance.
                                                                         Defaults to None.
            temporal_resolution (Optional[float], optional): The temporal resolution of the animation in seconds,
                                                             per default will be set same as the trajectory.
            plot_ax_ticks (Optional[bool], optional): Defaults to False.
            output (Optional[AnyStr], optional): Defaults to "network_graph_anim.mp4".
        """
        # shallow wrapper to plotting utils function
        self.multi_agent_net.animate_graph_net_trajectory(
            state_trajectory=self.state_trajectory,
            multi_agent_graph_seq=[G.G_communication for G in self.multi_agent_G_list],
            collision_communication_thrslds=collision_communication_thrslds,
            temporal_resolution=temporal_resolution,
            plot_ax_ticks=plot_ax_ticks,
            output=output,
            **kwargs,
        )

    def plot_network_graph_properties(
        self,
        func_used: Callable,
        ax: Optional[plt.axes] = None,
        figsize: Optional[Tuple[int]] = (8, 6),
        temporal_res: Optional[int] = 1800,
        xticks_temporal_res: Optional[int] = 14400,
    ) -> matplotlib.axes.Axes:
        """Generic function to be called with a function callable, defining which property of the graph should be plotted,
            e.g. the number of collisions, the number of isolated platforms etc.
        Args:
            func_used (Callable): function that plots the desired property, in the MultiAgent Class
                            - plot_isolated_vertices
                            - plot_collision_nb_over_time
                            - plot_graph_nb_connected_components
                            - plot_graph_degree
            ax (Optional[plt.axes], optional): Defaults to None.
            figsize (Optional[Tuple[int]], optional): Defaults to (8, 6).
            temporal_res (Optional[int], optional):  sampling interval for plotting from the trajectories in [s].
                                                    Defaults to 1800 (=30mins).
            xticks_temporal_res: (Optional[int], optional): xticks interval in [s]: Defaults to 14400 (= 4H).
        Returns:
            matplotlib.axes.Axes:
        """
        stride_temporal_res, stride_xticks = self.get_stride_for_xaxis_from_temp_res(
            temporal_res=temporal_res, xticks_temporal_res=xticks_temporal_res
        )
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.axes()
        dates = self.get_datetime_from_state_trajectory(state_trajectory=self.state_trajectory)
        func_used(
            ax=ax,
            list_of_graph=[G.G_communication for G in self.multi_agent_G_list],
            dates=dates,
            stride_temporal_res=stride_temporal_res,
            stride_xticks=stride_xticks,
        )
        return ax

    def plot_all_network_analysis(
        self,
        figsize: Optional[Tuple[int]] = (15, 15),
        temporal_res: Optional[int] = 1800,
        xticks_temporal_res: Optional[int] = 14400,
    ) -> matplotlib.figure.Figure:
        """Plot all the graph network properties over simulation time, such as communication losses, collisions etc.
        Serves as visual metric
        Args:
            figsize (Optional[Tuple[int]], optional): Defaults to (15, 15).
            temporal_res (Optional[int], optional): sampling interval for plotting from the trajectories in [s].
                                                    Defaults to 1800 (=30mins).
            xticks_temporal_res (Optional[int], optional): xticks interval in [s]: Defaults to 14400 (= 4H).

        Returns:
            matplotlib.figure.Figure
        """

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        ax1 = self.plot_network_graph_properties(
            func_used=self.multi_agent_net.plot_isolated_vertices,
            ax=ax1,
            temporal_res=temporal_res,
            xticks_temporal_res=xticks_temporal_res,
        )

        ax2 = self.plot_network_graph_properties(
            func_used=self.multi_agent_net.plot_collision_nb_over_time,
            ax=ax2,
            temporal_res=temporal_res,
            xticks_temporal_res=xticks_temporal_res,
        )
        ax3 = self.plot_network_graph_properties(
            func_used=self.multi_agent_net.plot_graph_nb_connected_components,
            ax=ax3,
            temporal_res=temporal_res,
            xticks_temporal_res=xticks_temporal_res,
        )
        ax4 = self.plot_network_graph_properties(
            func_used=self.multi_agent_net.plot_graph_degree,
            ax=ax4,
            temporal_res=temporal_res,
            xticks_temporal_res=xticks_temporal_res,
        )
        return fig

    def save_metrics_to_log(
        self, all_pltf_status: list, max_correction_from_opt_ctrl: list, filename: str
    ) -> dict:
        """Compute and save metrics for the given multi-agent instance
        For now implemented are:
        1) Time-integral metric of # isolated platforms
        2) Number of collisions during simulation
        3) The fraction of total platforms reaching the target (1 = all platforms reached the target within simulation time)
        4) The average of the maximum deviation from the optimal control angle for all the platforms
                                            at each simulation step as a proxy for energy efficiency
        5) The initial maximum degree of the graph
        6) The final maximum degree of the graph
        7) Mission success defined as if 3)= 1, 2) = 0 and 1) =0

        Args:
            all_pltf_status (list): list of platform status, obtained by calling problem_status through the simulation
                                    (see function above in arena)
            max_correction_from_opt_ctrl(list): maximum deviation from the optimal control angle for all the platforms
                                            at each simulation step
            filename (str): logging filename

        Returns:
            dict: dictionary form containing the different metrics
        """
        # success if platform has a problem status = 1 (target reached)
        success_rate_reach_target = sum(pltf_status == 1 for pltf_status in all_pltf_status) / len(
            all_pltf_status
        )
        energy_efficiency_proxy = np.mean(max_correction_from_opt_ctrl)
        metrics_dict = self.multi_agent_net.log_metrics(
            list_of_graph=[G.G_communication for G in self.multi_agent_G_list],
            dates=self.state_trajectory[0, 2, ::1],
            success_rate_reach_target=success_rate_reach_target,
            energy_efficiency_proxy=energy_efficiency_proxy,
            logfile=filename,
        )
        return metrics_dict
