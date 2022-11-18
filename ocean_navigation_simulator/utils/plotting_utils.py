from typing import AnyStr, Callable, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.utils.misc import get_markers
import networkx as nx

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


def plot_graph(G, pos):
    scaling = 1.05  # smaller numbers leads to larger distance representation when plotting
    weight_edges = nx.get_edge_attributes(G, "weight")
    weight_edges_for_scaling = {key: 1 / (value.km) for key, value in weight_edges.items()}
    edge_labels = {key: f"{value.km:.1f}" for key, value in weight_edges.items()}
    Gplot = G  # create a copy from G
    nx.set_edge_attributes(Gplot, values=weight_edges_for_scaling, name="weight")
    pos = nx.spring_layout(
        Gplot, seed=10, pos=pos, weight="weight"
    )  # edge length inversely prop to distance repr. on the graph
    ax = plt.axes()
    nx.draw_networkx(Gplot, pos=pos, with_labels=True)
    nx.draw_networkx_edge_labels(Gplot, pos, edge_labels=edge_labels)
    return ax


def animate_trajectory(
    state_trajectory: np.ndarray,
    ctrl_trajectory: np.ndarray,
    data_source: DataSource,
    margin: Optional[float] = 1,
    problem: Optional[NavigationProblem] = None,
    temporal_resolution: Optional[float] = None,
    add_ax_func_ext: Optional[Callable] = None,
    output: Optional[AnyStr] = "traj_animation.mp4",
    **kwargs,
):
    """Plotting functions to animate a trajectory including the controls over a data_source.
    Args:
          state_trajectory:  State trajectory as numpy array, first three rows need to be x, y, t
          ctrl_trajectory:   Control trajectory as numpy array, first two rows thrust_magnitude, angle (in radians)
          data_source:       DataSource object over which to animate the data (often an OceanCurrent source)
          ## Optional ##
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

    # Step 1: Define callable function of trajectory for each time to add on top of data_source plot
    def add_traj_and_ctrl_at_time(ax, time):
        # if there's a func plot it
        if add_ax_func_ext is not None:
            add_ax_func_ext(ax, time)
        markers = get_markers()
        # plot start position
        for k in range(state_trajectory.shape[0]):
            marker = next(markers)
            ax.scatter(
                state_trajectory[k, 0, 0],
                state_trajectory[k, 1, 0],
                c="r",
                marker=marker,
                s=200,
                label=f"Start for platform {k}",
            )
            ax.scatter(
                state_trajectory[k, 0, -1],
                state_trajectory[k, 1, -1],
                c="orange",
                marker=marker,
                s=200,
                label="Trajectory end" if k == 0 else "",
            )
            # add the trajectory to it
            ax.plot(
                state_trajectory[k, 0, :],
                state_trajectory[k, 1, :],
                color="black",
                linewidth=3,
                linestyle="--",
                label="State Trajectory" if k == 0 else "",
            )
        # plot the goal
        if problem is not None:
            goal_circle = plt.Circle(
                (problem.end_region.lon.deg, problem.end_region.lat.deg),
                problem.target_radius,
                color="g",
                fill=True,
                alpha=0.5,
                label="goal",
            )
            ax.add_patch(goal_circle)

        # plot the control arrow for the specific time
        for k in range(state_trajectory.shape[0]):
            # get the planned idx of current time
            idx = np.searchsorted(a=state_trajectory[k, 2, :], v=time)
            if (
                idx < state_trajectory.shape[2]
            ):  # if platforms did not start at the same time, some trajectories might finish sooner and there is no more control input
                ax.scatter(
                    state_trajectory[k, 0, idx],
                    state_trajectory[k, 1, idx],
                    c="m",
                    marker="o",
                    s=20,
                )
                ax.quiver(
                    state_trajectory[k, 0, idx],
                    state_trajectory[k, 1, idx],
                    ctrl_trajectory[k, 0, idx] * np.cos(ctrl_trajectory[k, 1, idx]),  # u_vector
                    ctrl_trajectory[k, 0, idx] * np.sin(ctrl_trajectory[k, 1, idx]),  # v_vector
                    color="magenta",
                    scale=5,
                    label="Control" if k == 0 else "",
                )
            else:
                continue
        ax.legend(loc="lower right", prop={"size": 8})

    # Step 2: Get the bounds for the data_source
    x_interval, y_interval, t_interval = get_lon_lat_time_interval_from_trajectory(
        state_trajectory=state_trajectory, margin=margin
    )

    # Step 3: run the animation with the data_source and the extra function
    data_source.animate_data(
        x_interval=x_interval,
        y_interval=y_interval,
        t_interval=t_interval,
        temporal_resolution=temporal_resolution,
        add_ax_func=add_traj_and_ctrl_at_time,
        output=output,
        **kwargs,
    )
