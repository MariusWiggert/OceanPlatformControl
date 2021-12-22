import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from IPython.display import HTML
from functools import partial
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timezone
from ocean_navigation_simulator.utils.simulation_utils import get_current_data_subset, convert_to_lat_lon_time_bounds
import warnings
import os


# Plotting utils for Ocean currents
def visualize_currents(time, grids_dict, u_data, v_data, vmin=0, vmax=None, alpha=0.5,
                       autoscale=False, plot=True, reset_plot=False, figsize=(12, 12)):
    """ Function to visualize ocean currents and optionally build visualization capabilities on top.
    Inputs:
        time                    time to plot the currents (timestamp in UTC POSIX)
        grids_dict              dict containing x_grid, y_grid, t_grid, fixed_time_idx
        u_data                  [T, Y, X] matrix of the ocean currents in x direction in m/s
        v_data                  [T, Y, X] matrix of the ocean currents in y direction in m/s

        Optional:
        vmin                    minimum current magnitude used for colorbar (float)
        vmax                    maximum current magnitude used for colorbar (float)
        alpha                   alpha of the current magnitude color visualization
        autoscale               True or False whether the plot is scaled to the figsize or lat-lon proportional
        plot                    if True: plt.show() is called directly, if False ax object is return to add things
        reset_plot              if True the current figure is resetted and no figure is created (used for animation)
        figsize                 size of the figure (per default (12,12))

    Outputs:
        ax object               if plot=False an ax object is returned to add further points/lines to the plot
    """
    print(time)

    # reset plot this is needed for matplotlib.animation
    if reset_plot:
        plt.clf()
    else:  # create a new figure object where it this is plotted
        fig = plt.figure(figsize=figsize)

    # Step 0: Create the figure and cartophy axis object and things (ocean, land-boarders,grid_lines, etc.)
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    grid_lines = ax.gridlines(draw_labels=True)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    ax.set_title("Time: " + datetime.fromtimestamp(time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))

    # Step 1: perform time-interpolation on the current data
    # Note: let's outsource that to utils? Especially when we do animation, makes sens to create it once.
    u_currents = interp1d(grids_dict['t_grid'], u_data, axis=0, kind='linear')(time)
    v_currents = interp1d(grids_dict['t_grid'], v_data, axis=0, kind='linear')(time)

    # get current vector magnitude in m/s
    magnitude = (u_currents ** 2 + v_currents ** 2) ** 0.5

    # Step 2: plot the current vectors in black
    X, Y = np.meshgrid(grids_dict['x_grid'], grids_dict['y_grid'], indexing='xy')
    plt.quiver(X, Y, u_currents, v_currents)

    # Step 3: underly the current magnitude in color
    if vmax is None:
        vmax = np.max(magnitude)
    if autoscale:
        plt.imshow(np.flip(magnitude, axis=0), extent=[grids_dict['x_grid'][0], grids_dict['x_grid'][-1],
                                      grids_dict['y_grid'][0], grids_dict['y_grid'][-1]],
                   aspect='auto',
                   cmap='jet', vmin=vmin, vmax=vmax, alpha=alpha)
    else:
        plt.imshow(np.flip(magnitude, axis=0), extent=[grids_dict['x_grid'][0], grids_dict['x_grid'][-1],
                                      grids_dict['y_grid'][0], grids_dict['y_grid'][-1]],
                   cmap='jet', vmin=vmin, vmax=vmax, alpha=alpha)

    # Step 4: set and format colorbar
    cbar = plt.colorbar()
    cbar.ax.set_title('current velocity')
    cbar.set_ticks(cbar.get_ticks())
    cbar.set_ticklabels(["{:.1f}".format(l) + ' m/s' for l in cbar.get_ticks().tolist()])

    # plot the figure
    if plot:
        plt.show()
    else:  # return ax object to draw other things on top of this
        return ax


def viz_current_animation(plot_times, grids_dict, u_data, v_data, ax_adding_func=None,
                          autoscale=False, interval=250, alpha=0.5, figsize=(12, 12),
                          save_as_filename=None, html_render=None):
    """ Function to animate ocean currents.
    Inputs:
        plot_times              vector of times to plot the currents (timestamps in POSIX)
        grids_dict              dict containing x_grid, y_grid, t_grid, fixed_time_idx
        u_data                  [T, Y, X] matrix of the ocean currents in x direction in m/s
        v_data                  [T, Y, X] matrix of the ocean currents in y direction in m/s

        Optional:
        ax_adding_func          function handle what to add on top of the current visualization
                                signature needs to be such that it takes an axis object and time as input
                                e.g. def add(ax, time, x=10,y=4): ax.scatter(x,y)
        alpha                   alpha of the current magnitude color visualization
        autoscale               True or False whether the plot is scaled to the figsize or lat-lon proportional
        interval                interval between frames in ms (for HTML directly in jupyter otherwise fps=10 is fixed)
        figsize                 size of the figure (per default (12,12))
        html_render             if None then html is displayed directly (for Jupyter)
                                if 'safari' it's opened in a safari page

    Outputs:
        file                    if save_as_filename is a string with 'name.gif' a gif will be created or for
                                'name.mp4' an mp4 file and saved in the currently active folder.
    """

    # get rounded up vmax across the whole dataset (with ` decimals)
    vmax = round(((u_data ** 2 + v_data ** 2) ** 0.5).max() + 0.049, 1)

    # create global figure object where the animation happens
    fig = plt.figure(figsize=(12, 12))

    # create full func for rendering the frame
    if ax_adding_func is not None:
        def full_plot_func(time, grids_dict, u_data, v_data, vmax, autoscale, alpha, figsize):
            # plot underlying currents at time
            ax = visualize_currents(time, grids_dict, u_data, v_data, autoscale=autoscale,
                                    plot=False, reset_plot=True, alpha=alpha, figsize=figsize, vmax=vmax)
            # add the ax_adding_func
            ax_adding_func(ax, time)

        # create partial func from the full_function
        render_frame = partial(full_plot_func,
                               grids_dict=grids_dict, u_data=u_data, v_data=v_data,
                               vmax=vmax, autoscale=autoscale, alpha=alpha, figsize=figsize)
    else:
        # create a partial function with most variables already set for the animation loop to call
        render_frame = partial(visualize_currents,
                               grids_dict=grids_dict, u_data=u_data, v_data=v_data,
                               vmax=vmax, autoscale=autoscale, plot=False, reset_plot=True,
                               alpha=alpha, figsize=figsize)

    # create animation function object (it's not yet executed)
    ani = animation.FuncAnimation(fig, func=render_frame, fargs=(), frames=plot_times,
                                  interval=interval, repeat=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if save_as_filename is None:
            ani_html = HTML(ani.to_html5_video())
            # Render using safari (only used because of Pycharm local testing)
            if html_render == 'safari':
                with open("current_animation.html", "w") as file:
                    file.write(ani_html.data)
                    os.system(
                        'open "/Applications/Safari.app" ' + '"' + os.path.realpath(
                            "current_animation.html") + '"')
            else: # visualize in Jupyter directly
                plt.close()
                return ani_html
        elif '.gif' in save_as_filename:
            ani.save(save_as_filename, writer=animation.PillowWriter(fps=10))
            plt.close()
        elif '.mp4' in save_as_filename:
            ani.save(save_as_filename, writer=animation.FFMpegWriter(fps=10))
            plt.close()
        else:
            raise ValueError("save_as_filename can be either None (for HTML rendering) or filepath and name needs to"
                             "contain either '.gif' or '.mp4' to specify the format and desired file location.")


def plot_land_mask(grids_dict):
    """Plot the land-mask of the current set."""
    if not np.all(grids_dict['spatial_land_mask']):
        plt.imshow(np.flip(grids_dict['spatial_land_mask'], axis=0),
                   extent=[grids_dict['x_grid'][0], grids_dict['x_grid'][-1],
                           grids_dict['y_grid'][0], grids_dict['y_grid'][-1]])
        plt.xlabel('longitude in deg')
        plt.ylabel('latitude in deg')
        plt.show()
    else:
        print("No land-mask, only open-ocean in the current data subset.")


def plot_2D_traj_over_currents(x_traj, time, file_dicts, x_T=None, x_T_radius=None, ctrl_seq=None, u_max=None):
    # Step 0: get respective data subset from hindcast file
    lower_left = [np.min(x_traj[0,:]), np.min(x_traj[1,:]), 0, time]
    upper_right = [np.max(x_traj[0, :]), np.max(x_traj[1, :])]

    t_interval, lat_bnds, lon_bnds = convert_to_lat_lon_time_bounds(lower_left, upper_right,
                                                                    deg_around_x0_xT_box=0.5,
                                                                    temp_horizon_in_h=5)
    grids_dict, u_data, v_data = get_current_data_subset(t_interval, lat_bnds, lon_bnds,
                                                         file_dicts=file_dicts)

    # define helper functions to add on top of current visualization
    def add_ax_func(ax, x_traj=x_traj):
        # plot trajectory
        ax.plot(x_traj[0, :], x_traj[1, :], '-', marker='x', c='k', linewidth=5)
        # plot start and end
        ax.scatter(x_traj[0,0], x_traj[1,0], c='r', marker='o', s=200, label='start')
        # ax.scatter(x_traj[0, -1], x_traj[1, -1], c='g', marker='x', s=200, label='end')
        if x_T is not None and x_T_radius is not None:
            goal_circle = plt.Circle((x_T[0], x_T[1]), x_T_radius, color='g', fill=True, alpha=0.5, label='goal')
            ax.add_patch(goal_circle)
        # plot control if supplied
        if ctrl_seq is not None:
            # calculate u and v direction
            u_vec = ctrl_seq[0,:] * np.cos(ctrl_seq[1,:])
            v_vec = ctrl_seq[0,:] * np.sin(ctrl_seq[1, :])
            ax.quiver(x_traj[0,:-1], x_traj[1,:-1], u_vec, v_vec, color='m', scale=15, label="u_max=" + str(u_max) + "m/s")
        plt.legend(loc='upper right')

    # plot underlying currents at time
    ax = visualize_currents(time, grids_dict, u_data, v_data, autoscale=True, plot=False)
    # add the start and goal position to the plot
    add_ax_func(ax)
    plt.show()


def plot_2D_traj_animation(traj_full, control_traj, file_dicts, u_max, html_render=None, filename=None):
    # extract space and time trajs
    space_traj = traj_full[:2,:]
    traj_times = traj_full[3,:]

    # Step 0: get respective data subset from hindcast file
    lower_left = [np.min(space_traj[0,:]), np.min(space_traj[1,:]), 0, traj_times[0]]
    upper_right = [np.max(space_traj[0, :]), np.max(space_traj[1, :])]

    t_interval, lat_bnds, lon_bnds = convert_to_lat_lon_time_bounds(lower_left, upper_right,
                                                                    deg_around_x0_xT_box=0.5,
                                                                    temp_horizon_in_h=(traj_times[-1]-traj_times[0])/3600)
    grids_dict, u_data, v_data = get_current_data_subset(t_interval, lat_bnds, lon_bnds, file_dicts=file_dicts)

    # define helper functions to add on top of current visualization
    def add_ax_func(ax, time):
        # plot start position
        ax.scatter(space_traj[0, 0], space_traj[1, 0], c='r', marker='o', s=200, label='start')
        ax.scatter(space_traj[0, -1], space_traj[1, -1], c='g', marker='*', s=200, label='end')
        # plot a dot at current position (assumes same times as plotting)
        idx = np.where(traj_times == time)[0][0]
        # plot actuation vector & resulting vector after vector_addition
        if idx < control_traj.shape[1]:
            # calculate u and v direction
            u_vec = control_traj[0,idx]*np.cos(control_traj[1, idx])
            v_vec = control_traj[0,idx]*np.sin(control_traj[1, idx])
            ax.quiver(space_traj[0, idx], space_traj[1, idx], u_vec, v_vec, color='m', scale=10, label="u_max=" + str(u_max))
        # plot full line
        ax.plot(space_traj[0, :], space_traj[1, :], '--', marker='x', c='k', linewidth=1)
        plt.legend(loc='upper right')
        ax.scatter(space_traj[0, idx], space_traj[1, idx], c='m', marker='o', s=50)

    # plot with extra function
    viz_current_animation(traj_times, grids_dict, u_data, v_data, interval=200, ax_adding_func=add_ax_func,
                          html_render=html_render, save_as_filename=filename)


def plot_2D_traj(x_traj, title="Planned Trajectory"):
    plt.figure(1)
    plt.plot(x_traj[0, :], x_traj[1, :], '--', marker='o')
    plt.plot(x_traj[0, 0], x_traj[1, 0], '--', marker='x', color='red')
    plt.plot(x_traj[0, -1], x_traj[1, -1], '--', marker='x', color='green')
    plt.title(title)
    plt.xlabel('lon')
    plt.ylabel('lat')
    # plt.grid()
    plt.show()


def plot_opt_ctrl(times, ctrl_seq, title='Planned Optimal Control'):
    plt.figure(2)
    fig, ax = plt.subplots(1, 1)
    # some stuff for flexible date axis
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # plot
    dates = [datetime.fromtimestamp(posix, tz=timezone.utc) for posix in times]
    ax.step(dates, ctrl_seq[0, :], where='post', label='u_power')
    ax.step(dates, ctrl_seq[1, :], where='post', label='angle')
    plt.title(title)
    plt.ylabel('u_power and angle in units')
    plt.xlabel('time')
    # plt.grid()
    plt.show()


def plot_opt_results(T,u,x,N):
    plt.figure(3)
    plt.plot(np.linspace(0., T, N), u[0, :], '-.')
    plt.plot(np.linspace(0., T, N), u[1, :], '-.')
    plt.title('Results from ipopt Optimization')
    plt.xlabel('time in h')
    plt.ylabel('Actuation velocity in m/s')
    plt.legend(['u_x trajectory', 'u_y trajectory'])
    # plt.grid()
    plt.show()
    plot_2D_traj(x)
    return



