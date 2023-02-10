import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd


def plot_missions_target_cartopy(df: pd.DataFrame, region: str):
    if region == "GOM":
        x_range = [-98, -76]
        y_range = [18, 32]
    elif region == "region 3":
        x_range = [-120, -70]
        y_range = [-20, 20]
    else:
        return "region not yet implemented"
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_title("Time: " + datetime.fromtimestamp(time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))
    ax.add_feature(cfeature.LAND, zorder=3, edgecolor="black")
    ax.add_feature(cfeature.OCEAN, zorder=0)
    grid_lines = ax.gridlines(draw_labels=True, zorder=5)
    grid_lines.top_labels = False
    grid_lines.right_labels = False
    ax.axis(xmin=x_range[0], xmax=x_range[1])
    ax.axis(ymin=y_range[0], ymax=y_range[1])
    for miss_idx in range(len(df)):
        ax.scatter(
            df.iloc[miss_idx]["x_0_lon"],
            df.iloc[miss_idx]["x_0_lat"],
            c="red",
            marker="o",
            s=6,
            label="starts" if miss_idx == 0 else None,
        )
    ax.scatter(df["x_T_lon"], df["x_T_lat"], c="green", marker="x", s=12, label="targets")
    ax.legend()
    # ax.get_figure().savefig(f"{analysis_folder}starts_and_targets.png")
    ax.get_figure().show()
