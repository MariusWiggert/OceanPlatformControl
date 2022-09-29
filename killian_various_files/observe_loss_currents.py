import os

import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

from ocean_navigation_simulator.environment.data_sources.DataSources import DataSource
from ocean_navigation_simulator.environment.data_sources.OceanCurrentSource.OceanCurrentSource import OceanCurrentSource
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentRunner import compute_conservation_mass_loss
from ocean_navigation_simulator.utils import units

print(os.getcwd())
# %%
folder = "forecast_c3_script/hc_may/"
files = os.listdir(folder)
xr_hindcast = xr.Dataset()
for file in sorted(files)[:4]:
    print(file)
    data = xr.open_dataset(folder + file).sel(lon=slice(-94, -84), lat=slice(22.5, 27.5))
    xr_hindcast = xr_hindcast.combine_first(data)
xr_hindcast = xr_hindcast.isel(depth=0).drop_vars('depth')
print(xr_hindcast)

# %%
# print(merged.isel(time=0).to_array("water_u"))
# f, axes = plt.subplots(2)
# OceanCurrentSource.plot_data_from_xarray(0, merged, ax=axes[0])
# merged.isel(time=0).water_u.plot()

# %%
# Add a dimension corresponding to the batch dimension
tensor_merged = torch.tensor(xr_hindcast.to_array().to_numpy())[None, :]
# %%
loss, res = compute_conservation_mass_loss(tensor_merged, get_all_cells=True)
res = res[0]
print("total_loss", loss)
print(res.shape)
# %%
padded_res = np.pad(res.numpy(), ((0, 0), (1, 0), (1, 0)))
print(padded_res.shape, xr_hindcast.to_array().shape[-3:])
assert padded_res.shape == xr_hindcast.to_array().shape[-3:]
xr_loss = xr.DataArray(padded_res, coords=xr_hindcast.coords).to_dataset(name="conservation_loss")


# %%
# DataSource.plot_data_from_xarray(0, xr_loss, ax=axes[1])


# %%
def visualize_initial_error(xr_1, xr_2, radius_area: float = None, ):
    # fig, ax = plt.subplots(2, 2)
    # ax1, ax2, ax3, ax4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    fig, axes = plt.subplots(2)
    ax1, ax2 = axes[0], axes[1]

    global cbar, cbar_2
    ax1, cbar = OceanCurrentSource.plot_data_from_xarray(0, xr_1, ax=ax1,
                                                         return_cbar=True)
    ax2, cbar_2 = DataSource.plot_data_from_xarray(0, xr_2, ax=ax2, return_cbar=True)

    def update_maps(lag, index_prediction, ax1, ax2):
        global cbar, cbar_2
        index_prediction *= 24
        ax1.clear()
        ax2.clear()
        cbar.remove()
        cbar_2.remove()
        fig.suptitle('At time {}, with a lag of {} hours.'.format(
            units.get_datetime_from_np64(xr_1.isel(time=slice(index_prediction, index_prediction + 24))['time'][0]),
            lag), fontsize=14)
        _, cbar = OceanCurrentSource.plot_data_from_xarray(lag, xr_1.isel(
            time=slice(index_prediction, index_prediction + 24)), ax=ax1,
                                                           return_cbar=True)
        _, cbar_2 = DataSource.plot_data_from_xarray(lag,
                                                     xr_2.isel(time=slice(index_prediction, index_prediction + 24)),
                                                     ax=ax2, return_cbar=True)
        ax1.set_title("Hindcast [m/s]")
        ax2.set_title("loss [m/s]")

    time_dim = list(range(len(xr_1["time"])))

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    ax_lag_time = plt.axes([0.25, 0.1, 0.65, 0.03])
    lag_time_slider = Slider(
        ax=ax_lag_time,
        label='Lag (in hours) with hindcast',
        valmin=min(time_dim),
        valmax=23,
        valinit=time_dim[0],
        valstep=1
    )

    # Make a vertically oriented slider to control the amplitude
    axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
    forecast_slider = Slider(
        ax=axamp,
        label="Day Hindcast",
        valmin=0,
        valmax=len(xr_1.time) // 24 - 1,
        valinit=0,
        orientation="vertical",
        valstep=1
    )

    # The function to be called anytime a slider's value changes
    def update(_):
        update_maps(lag_time_slider.val, forecast_slider.val, ax1, ax2)
        fig.canvas.draw_idle()

    # register the update function with each slider
    lag_time_slider.on_changed(update)
    forecast_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        ax_lag_time.reset()
        forecast_slider.reset()

    button.on_clicked(reset)
    update_maps(0, 0, ax1, ax2)
    plt.show()
    keyboardClick = False
    while keyboardClick != True:
        keyboardClick = plt.waitforbuttonpress()


# %%
visualize_initial_error(xr_hindcast, xr_loss)
# %% Test the loss function
a1, a2, a3, a4, a5, a6, = [10, 10], [10, 10], [10, 10], [10, 10], [10, 10], [10, 10]
pred = torch.moveaxis(torch.tensor([[[a1, a2, a3], [a4, a5, a6]]], dtype=torch.double), -1, 1)[:, :, None]
print(pred.shape)
_, all = compute_conservation_mass_loss(pred, get_all_cells=True)
all
#
