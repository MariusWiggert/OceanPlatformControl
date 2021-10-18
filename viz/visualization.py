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

        f.set_figwidth(size)
        f.set_figheight(size)

        pc = ccrs.PlateCarree()
        ax = plt.axes(projection=pc)
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')

        y_grid = self.grids_dict.get("y_grid")
        y_grid = np.flip(y_grid, axis=0)
        u_data = np.flip(self.u_data, axis=0)
        v_data = np.flip(self.v_data, axis=0)

        X, Y = np.meshgrid(self.grids_dict.get("x_grid"), y_grid, indexing='xy')

        start = datetime.utcfromtimestamp(self.grids_dict['t_grid'][time_idx]).strftime('%Y-%m-%d %H:%M:%S UTC')
        ax.set_title(start)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        magnitude = (u_data[time_idx, ...] ** 2 + v_data[time_idx, ...] ** 2) ** 0.5
        ax.set_xticks([])  # workaround for setting axes labels
        ax.set_yticks([])  # workaround for setting adxes labels

        colormap = cm.rainbow
        norm = Normalize()
        norm.autoscale(magnitude)

        mapped_colors = colormap(norm(magnitude))
        for i in range(len(mapped_colors)):
            for j in range(len(mapped_colors[0])):
                num = np.average(mapped_colors[i][j])  # average the magnitude to support color length parameter
                mapped_colors[i][j][3] = num

        ax.quiver(X, Y, u_data[time_idx, ...], v_data[time_idx, ...], mapped_colors[..., 3])

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
        os.mkdir('temp_photos')
        for i in range(len(self.u_data)):
            self.visualize(time_idx=i, animate=True)
        png_dir = 'temp_photos'
        images = []
        for file_name in sorted(os.listdir(png_dir)):
            if file_name.endswith('.png'):
                try:
                    file_path = os.path.join(png_dir, file_name)
                    images.append(imageio.imread(file_path))
                except (IOError, SyntaxError) as e:
                    print('Bad file:', file_name)
        imageio.mimsave('gifs/' + self.plot_name + '.gif', images)
        shutil.rmtree('temp_photos')
