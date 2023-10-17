import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def plot_monthly_means(data, lon_range, lat_range, save_path: str = ""):
    res = [*data["U"].T.shape]
    lon_res = np.linspace(-180, 180, res[1])
    lat_res = np.linspace(-75, 85, res[0])

    pix_idx_lon = [
        min(range(len(lon_res)), key=lambda i: abs(lon_res[i] - lon_range[0])),
        min(range(len(lon_res)), key=lambda i: abs(lon_res[i] - lon_range[1])),
    ]
    pix_idx_lat = [
        min(range(len(lat_res)), key=lambda i: abs(lat_res[i] - lat_range[0])),
        min(range(len(lat_res)), key=lambda i: abs(lat_res[i] - lat_range[1])),
    ]

    region_data = []
    magnitude = np.sqrt(data["U"] ** 2 + data["V"] ** 2)
    for time in range(12):
        region_data.append(
            magnitude.sel(
                longitude=slice(pix_idx_lon[0], pix_idx_lon[1]),
                latitude=slice(pix_idx_lat[0], pix_idx_lat[1]),
                time=time,
            ).T
        )

    # plotting
    fig, axs = plt.subplots(4, 3, figsize=(18, 20))
    x_tick_labels = np.linspace(
        lon_range[0], lon_range[1], round((lon_range[1] - lon_range[0]) / (res[1] / 360))
    )
    y_tick_labels = np.linspace(
        lat_range[0], lat_range[1], round((lat_range[1] - lat_range[0]) / (res[0] / 180))
    )
    if len(x_tick_labels) > 8:
        x_tick_labels = np.linspace(lon_range[0], lon_range[1], 8)
    if len(y_tick_labels) > 8:
        y_tick_labels = np.linspace(lat_range[0], lat_range[1], 8)
    x_ticks = np.linspace(0, abs(pix_idx_lon[1] - pix_idx_lon[0]), len(x_tick_labels))
    y_ticks = np.linspace(0, abs(pix_idx_lat[0] - pix_idx_lat[1]), len(y_tick_labels))

    # round labels
    x_tick_labels = [round(label, 1) for label in x_tick_labels]
    y_tick_labels = [round(label, 1) for label in y_tick_labels]

    month_list = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    for idx in range(0, 12):
        axis = axs[idx // 3, idx % 3]
        frame = axis.pcolormesh(region_data[idx])
        axis.set_xlabel(f"{month_list[idx]}")
        axis.set_xticks(x_ticks)
        axis.set_yticks(y_ticks)
        axis.set_xticklabels(x_tick_labels)
        axis.set_yticklabels(y_tick_labels)
        plt.colorbar(frame, ax=axis)
    # plt.suptitle("Monthly average current magnitude for GoM", size=30, va="top")
    plt.tight_layout(pad=1.8)
    plt.show()

    if not save_path == "":
        fig.savefig(save_path)


if __name__ == "__main__":
    # data from: https://www.aoml.noaa.gov/phod/gdp/mean_velocity.php
    data = xr.open_dataset("/home/jonas/Downloads/drifter_monthlymeans.nc", decode_times=False)
    # data_total = data["U"].values[0]
    # plt.pcolormesh(data_total.T)
    # plt.show()

    # # My Region 1
    # lon_range = np.array([-146.25, -125])
    # lat_range = np.array([15, 36.25])

    # # Region 1
    # lon_range = np.array([-160, -105])
    # lat_range = np.array([10, 65])

    # GoM
    lon_range = np.array([-98, -80])
    lat_range = np.array([16, 33])

    # # Region 6
    # lon_range = np.array([-36, 16])
    # lat_range = np.array([-33, 6])

    plot_monthly_means(data, lon_range, lat_range, "/home/jonas/Downloads/GoM_trends.png")
