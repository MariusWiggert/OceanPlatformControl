import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def plot_monthly_means(data, lon_range, lat_range):
    res = [*data["U"].T.shape]
    lon_res = np.linspace(-180, 180, res[1])
    lat_res = np.linspace(-90, 90, res[0])

    pix_idx_lon = [min(range(len(lon_res)), key=lambda i: abs(lon_res[i] - lon_range[0])),
                   min(range(len(lon_res)), key=lambda i: abs(lon_res[i] - lon_range[1]))]
    pix_idx_lat = [min(range(len(lat_res)), key=lambda i: abs(lat_res[i] - lat_range[0])),
                   min(range(len(lat_res)), key=lambda i: abs(lat_res[i] - lat_range[1]))]

    region_data = []
    magnitude = np.sqrt(data["U"]**2 + data["V"]**2)
    for time in range(12):
        region_data.append(magnitude.sel(longitude=slice(pix_idx_lon[0], pix_idx_lon[1]),
                                           latitude=slice(pix_idx_lat[0], pix_idx_lat[1]),
                                           time=time).T)

    # plotting
    fig, axs = plt.subplots(6, 2, figsize=(8, 20))
    x_tick_labels = np.linspace(lon_range[0], lon_range[1], round((lon_range[1]-lon_range[0])/(res[1]/360)))
    x_ticks = np.linspace(0, abs(pix_idx_lon[1] - pix_idx_lon[0]), len(x_tick_labels))
    y_tick_labels = np.linspace(lat_range[0], lat_range[1], round((lat_range[1]-lat_range[0])/(res[0]/180)))
    y_ticks = np.linspace(0, abs(pix_idx_lat[0] - pix_idx_lat[1]), len(y_tick_labels))

    month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for idx in range(0, 12):
        axis = axs[idx // 2, idx % 2]
        frame = axis.pcolormesh(region_data[idx])
        axis.set_xlabel(f"{month_list[idx]}")
        axis.set_xticks(x_ticks)
        axis.set_yticks(y_ticks)
        axis.set_xticklabels(x_tick_labels)
        axis.set_yticklabels(y_tick_labels)
        plt.colorbar(frame, ax=axis)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = xr.open_dataset("/home/jonas/Downloads/drifter_monthlymeans.nc", decode_times=False)

    # Region 1
    lon_range = np.array([-140, -120])
    lat_range = np.array([15, 35])

    # # GoM
    # lon_range = np.array([-100, -80])
    # lat_range = np.array([14, 30])

    plot_monthly_means(data, lon_range, lat_range)


