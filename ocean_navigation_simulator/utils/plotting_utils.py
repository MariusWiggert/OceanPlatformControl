from typing import AnyStr, Callable, List, Optional, Tuple
import plotly.io as pio
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.utils.misc import get_markers

idx_state = {
    "lon": 0,
    "lat": 1,
    "time": 2,
}


def get_lon_lat_time_interval_from_trajectory(
    state_trajectory: np.ndarray,
    margin: Optional[float] = 1,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Helper function to find the interval around start/trajectory/goal.
    Args:
        state_trajectory:   State Trajectory with rows x, y, t
        margin:             Optional[float]

    Returns:
        lon_interval:  [x_lower, x_upper] in degrees
        lat_interval:  [y_lower, y_upper] in degrees
        time_interval: [t_lower, t_upper] in posix time
    """
    # TODO: update comments:
    # now state_trajectory [#nb_platforms, #states, time]
    lon_min = np.min(state_trajectory[:, idx_state["lon"], :])
    lon_max = np.max(state_trajectory[:, idx_state["lon"], :])
    lat_min = np.min(state_trajectory[:, idx_state["lat"], :])
    lat_max = np.max(state_trajectory[:, idx_state["lat"], :])

    return (
        [lon_min - margin, lon_max + margin],
        [lat_min - margin, lat_max + margin],
        [
            np.min(state_trajectory[:, idx_state["time"], 0]),
            np.max(state_trajectory[:, idx_state["time"], -1]),
        ],
    )


def get_index_from_posix_time(state_trajectory: np.ndarray, posix_time: float) -> int:
    """Helper function to find the closest trajectory index corresponding to a given posix time.
    Args:
        state_trajectory:   State Trajectory with rows x, y, t
        posix_time: float

    Returns:
        index: float
    """
    return np.searchsorted(a=state_trajectory[2, :], v=posix_time)


def animate_trajectory(
    state_trajectory: np.ndarray,
    ctrl_trajectory: np.ndarray,
    data_source: DataSource,
    margin: Optional[float] = 1,
    x_interval: Optional[List[float]] = None,
    y_interval: Optional[List[float]] = None,
    problem: Optional[NavigationProblem] = None,
    temporal_resolution: Optional[float] = None,
    add_ax_func_ext: Optional[Callable] = None,
    output: Optional[AnyStr] = "traj_animation.mp4",
    full_traj: Optional[bool] = True,
    **kwargs,
):
    """Plotting functions to animate a trajectory including the controls over a data_source.
    Args:
          state_trajectory:  State trajectory as numpy array, first three rows need to be x, y, t
          ctrl_trajectory:   Control trajectory as numpy array, first two rows thrust_magnitude, angle (in radians)
          data_source:       DataSource object over which to animate the data (often an OceanCurrent source)
          ## Optional ##
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

    # Step 1: Define callable function of trajectory for each time to add on top of data_source plot
    def add_traj_and_ctrl_at_time(ax, time):
        # if there's a func plot it
        if add_ax_func_ext is not None:
            add_ax_func_ext(ax, time)
        # markers = get_markers() if different markers wished for each platform
        # plot start position
        for k in range(state_trajectory.shape[0]):
            idx = min(
                np.searchsorted(a=state_trajectory[k, 2, :], v=time), ctrl_trajectory.shape[2] - 1
            )
            if full_traj:
                traj_to_plot = state_trajectory[k, :, :]
                # marker = next(markers)
                ax.scatter(
                    state_trajectory[k, 0, -1],
                    state_trajectory[k, 1, -1],
                    c="black",
                    marker="*",  # marker,if different markers wished for each platform
                    s=150,
                    label="Trajectories end" if k == 0 else None,
                    zorder=5,
                )
            else:
                idx = np.searchsorted(a=state_trajectory[k, 2, :], v=time)
                traj_to_plot = state_trajectory[k, :, (idx + 1)]
            # add the trajectory to it
            ax.plot(
                traj_to_plot[0, :],
                traj_to_plot[1, :],
                color=kwargs.get("traj_color", "black"),
                linewidth=kwargs.get("traj_linewidth", 3),
                linestyle=kwargs.get("traj_linestyle", "--"),
                label="State Trajectory" if k == 0 else "",
                zorder=5,
            )
            ax.scatter(
                traj_to_plot[0, 0],
                traj_to_plot[1, 0],
                c="r",
                marker="o",  # marker,if different markers wished for each platform
                s=150,
                label="Start for platforms" if k == 0 else None,
                zorder=6,
            )
            # plot the control arrow for the specific time
            ax.scatter(
                state_trajectory[k, 0, idx],
                state_trajectory[k, 1, idx],
                c=kwargs.get("x_t_marker_color", "m"),
                marker="o",
                s=kwargs.get("x_t_marker_size", 20),
                zorder=7,
            )
            ax.quiver(
                state_trajectory[k, 0, idx],
                state_trajectory[k, 1, idx],
                ctrl_trajectory[k, 0, idx] * np.cos(ctrl_trajectory[k, 1, idx]),  # u_vector
                ctrl_trajectory[k, 0, idx] * np.sin(ctrl_trajectory[k, 1, idx]),  # v_vector
                color=kwargs.get("ctrl_color", "magenta"),
                scale=kwargs.get("ctrl_scale", 5),
                label="Control" if k == 0 else "",
                zorder=8,
            )
        # plot the goal
        if problem is not None:
            goal_circle = plt.Circle(
                (problem.end_region.lon.deg, problem.end_region.lat.deg),
                problem.target_radius,
                facecolor="None",
                linewidth=4,
                # facecolor=problem_target_color,
                edgecolor="g",
                zorder=8,
                label="Goal",
            )
            #     color="g",
            #     fill=True,
            #     alpha=0.7,
            #     label="Goal",
            #     zorder=6,
            # )
            ax.add_patch(goal_circle)
        ax.legend(loc="lower right", prop={"size": 8})

    # Step 2: Get the bounds for the data_source
    x_interval_margin, y_interval_margin, t_interval = get_lon_lat_time_interval_from_trajectory(
        state_trajectory=state_trajectory, margin=margin
    )

    # Step 3: run the animation with the data_source and the extra function
    data_source.animate_data(
        x_interval=x_interval if x_interval is not None else x_interval_margin,
        y_interval=y_interval if y_interval is not None else y_interval_margin,
        t_interval=t_interval,
        temporal_resolution=temporal_resolution,
        add_ax_func=add_traj_and_ctrl_at_time,
        output=output,
        **kwargs,
    )


def set_palatino_font(font_path="/package_data/font/Palatino_thin.ttf"):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = prop.get_name()
    params = {
        "legend.fontsize": "x-large",
        "axes.labelsize": 21,
        "axes.titlesize": 21,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    }
    plt.rcParams.update(params)


def set_palatino_font_plotly(font_path="/package_data/font/Palatino_thin.ttf"):
    # Load font
    pio.templates.default = "plotly"
    pio.templates[pio.templates.default]["layout"]["font"]["family"] = "Palatino"

    # Set font properties
    font = {"family": "Palatino", "size": 21}

    layout = go.Layout(
        font=font,
        legend=dict(
            font=dict(
                family=font["family"],
                size=25,
            ),
        ),
        title=dict(
            font=dict(
                family=font["family"],
                size=30,
            ),
        ),
        xaxis=dict(
            title_font=dict(
                family=font["family"],
                size=25,
            ),
            tickfont=dict(
                family=font["family"],
                size=13,
            ),
        ),
        yaxis=dict(
            title_font=dict(
                family=font["family"],
                size=25,
            ),
            tickfont=dict(
                family=font["family"],
                size=13,
            ),
        ),
    )

    return layout
