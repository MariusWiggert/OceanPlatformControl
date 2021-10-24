import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from datetime import datetime
import os
import shutil
import imageio

class Visualization:
    """
    Class to visualize currents on longitude and latitude axes. Ocean current vectors are colored by magnitude.
    grid_dicts, u_data, v_data = matrices obtained by get_current_data_subset method.
    x_0 = [initial longitude, initial latitude, ...]
    x_T = [destination longitude, destination latitude]
    nc_file_path = current only function serves as the name of the gif to be generated
    size = size to be passed into the figure size

    Example Usage:

    from src.utils.simulation_utils import get_current_data_subset

    hindcast_file = 'notebook_example/2021_06_1-05_hourly.nc4'
    x_0 = [-88.0, 25.0, 1, 1622549410.0]  # lon, lat, battery, posix_time
    x_T = [-88.0, 26.3]
    deg_around_x0_xT_box = 0.5
    fixed_time = None
    temporal_stride = 1

    grids_dict, u_data, v_data = get_current_data_subset(hindcast_file,
                                                         x_0, x_T,
                                                         deg_around_x0_xT_box,
                                                         fixed_time,
                                                         temporal_stride)

    viz = Visualization(grids_dict, u_data, v_data)
    viz.visualize()
    viz.animate()
    """
    def __init__(self, grids_dict, u_data, v_data, x_0=None, x_T=None, nc_file_path='Figure', size=12):
        self.grids_dict = grids_dict
        self.u_data = u_data
        self.v_data = v_data
        self.x_0 = x_0
        self.x_T = x_T
        self.plot_name = nc_file_path
        self.size = size

    def visualize(self, time_idx=0, animate=False):
        f = plt.figure()

        f.set_figwidth(self.size)
        f.set_figheight(self.size)

        pc = ccrs.PlateCarree()
        ax = plt.axes(projection=pc)
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
        ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

        y_grid = self.grids_dict.get("y_grid")
        y_grid = np.flip(y_grid, axis=0)
        u_data = np.flip(self.u_data, axis=0)
        v_data = np.flip(self.v_data, axis=0)

        X, Y = np.meshgrid(self.grids_dict.get("x_grid"), y_grid, indexing='xy')

        start = datetime.utcfromtimestamp(self.grids_dict['t_grid'][time_idx]).strftime('%Y-%m-%d %H:%M:%S UTC')
        ax.set_title(start)
        magnitude = (u_data[time_idx, ...] ** 2 + v_data[time_idx, ...] ** 2) ** 0.5


        colormap = cm.rainbow
        norm = Normalize()
        mapped_colors = colormap(norm(magnitude))
        norm.autoscale(mapped_colors)

        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        ax.quiver(X, Y, u_data[time_idx, ...], v_data[time_idx, ...])
        plt.colorbar(sm)
        I = plt.imshow(mapped_colors, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], cmap=colormap, norm=norm)

        if self.x_0 is not None and self.x_T is not None:
            plt.scatter([self.x_0[0], self.x_T[0]], [self.x_0[1], self.x_T[1]])

        if animate:
            time_str = str(time_idx)
            if len(time_str) == 2:
                time_str = '0' + time_str
            elif len(time_str) == 1:
                time_str = '00' + time_str
            plt.savefig('temp_photos/' + time_str + '.png')
            plt.close()
        else:
            plt.show()

    def animate(self):
        if not os.path.isdir('gifs'):
            os.mkdir('gifs')
        if os.path.isdir('temp_photos'):
            shutil.rmtree('temp_photos')
        png_dir = 'temp_photos'
        os.mkdir(png_dir)
        for i in range(len(self.u_data)):
            self.visualize(time_idx=i, animate=True)
        print('Done visualizing data over ' + str(len(self.u_data)) + ' images')
        images = []
        for file_name in sorted(os.listdir(png_dir)):
            if file_name.endswith('.png'):
                try:
                    file_path = os.path.join(png_dir, file_name)
                    images.append(imageio.imread(file_path))
                except (IOError, SyntaxError) as e:
                    print('Bad file:', file_name)
        imageio.mimsave('gifs/' + self.plot_name + '.gif', images)
        print('Done animating data in gifs/')
        shutil.rmtree('temp_photos')