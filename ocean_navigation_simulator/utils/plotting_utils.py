from typing import AnyStr, Callable, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from ocean_navigation_simulator.data_sources.DataSource import DataSource
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)


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
    lon_min = np.min(state_trajectory[0, :])
    lon_max = np.max(state_trajectory[0, :])
    lat_min = np.min(state_trajectory[1, :])
    lat_max = np.max(state_trajectory[1, :])

    return (
        [lon_min - margin, lon_max + margin],
        [lat_min - margin, lat_max + margin],
        [state_trajectory[2, 0], state_trajectory[2, -1]],
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
    problem: Optional[NavigationProblem] = None,
    temporal_resolution: Optional[float] = None,
    add_ax_func_ext: Optional[Callable] = None,
    output: Optional[AnyStr] = "traj_animation.mp4",
    **kwargs
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
        # plot start position
        ax.scatter(
            state_trajectory[0, 0], state_trajectory[1, 0], c="r", marker="o", s=200, label="Start"
        )
        ax.scatter(
            state_trajectory[0, -1],
            state_trajectory[1, -1],
            c="orange",
            marker="*",
            s=200,
            label="Traj_end",
        )
        # add the trajectory to it
        ax.plot(
            state_trajectory[0, :],
            state_trajectory[1, :],
            color="black",
            linewidth=2,
            linestyle="--",
            label="State Trajectory",
        )
        # plot the goal
        if problem is not None and hasattr(problem, "end_region"):
            goal_circle = plt.Circle(
                (problem.end_region.lon.deg, problem.end_region.lat.deg),
                problem.target_radius,
                color="g",
                fill=True,
                alpha=0.5,
                label="goal",
            )
            ax.add_patch(goal_circle)
        # get the planned idx of current time
        idx = np.searchsorted(a=state_trajectory[2, :], v=time)
        # plot the control arrow for the specific time
        ax.scatter(state_trajectory[0, idx], state_trajectory[1, idx], c="m", marker="o", s=20)
        ax.quiver(
            state_trajectory[0, idx],
            state_trajectory[1, idx],
            ctrl_trajectory[0, idx] * np.cos(ctrl_trajectory[1, idx]),  # u_vector
            ctrl_trajectory[0, idx] * np.sin(ctrl_trajectory[1, idx]),  # v_vector
            color="magenta",
            scale=10,
            label="Control",
        )
        ax.legend(loc="upper right")

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
        **kwargs
    )
