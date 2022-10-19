##
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

#

full_path = Path("ablation_study/configs_GP/gm_GP_025_12.yaml")
arena = ArenaFactory.create(scenario_name=full_path.resolve().stem, folder_scenario=full_path.parent)
fc = arena.ocean_field.forecast_data_source
hc = arena.ocean_field.hindcast_data_source  # .DataArray.interp_like(arena.ocean_field.forecast_data_source.DataArray)
j = 50
lat_interv = [22.0, 27]  # fc.DataArray['lat'][[j, -j]]
lon_interv = [-94, -86]  # [-95.362841, -85.766062]  # fc.DataArray['lon'][[j, -j]]
margin_space = 1 / 12
lat_interv_hc = (lat_interv[0] - margin_space, lat_interv[1] + margin_space)
lon_interv_hc = (lon_interv[0] - margin_space, lon_interv[1] + margin_space)

# july
start_july = datetime.datetime(2022, 7, 7, 12, 30, tzinfo=datetime.timezone.utc) + datetime.timedelta(hours=1)
# september
start_sept = datetime.datetime(2022, 9, 5, 12, 30, tzinfo=datetime.timezone.utc) + datetime.timedelta(hours=1)

for start in [start_july, start_sept]:
    delta = datetime.timedelta(hours=24)
    fig, axs = plt.subplots(2)
    data_fc = fc.get_data_over_area(lon_interv, lat_interv, [start, start + delta])
    fc_frame = data_fc.sel(time=start).assign(magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5)
    data_hc = hc.get_data_over_area(lon_interv_hc, lat_interv_hc, [start, start + delta])
    data_hc = data_hc.interp_like(data_fc)
    data_hc_one_time = data_hc.sel(time=start).assign(magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5)
    vmin = min(np.min(fc_frame['magnitude'].min()).item(), np.min(data_hc_one_time['magnitude'].min()).item())
    vmax = max(np.max(fc_frame['magnitude'].max()).item(), np.max(data_hc_one_time['magnitude'].max()).item())

    print("plot fc")
    # fc.plot_data_at_time_over_area(start, lon_interv, lat_interv, ax=axs[0])
    fc.plot_data_from_xarray(0, fc_frame, ax=axs[0], vmin=vmin, vmax=vmax)
    print("plot hc")
    hc.plot_data_from_xarray(0, data_hc_one_time, ax=axs[1], colorbar=False, vmin=vmin, vmax=vmax,
                             fill_space_for_cbar=False)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[0].set_title("Forecast - " + axs[0].title.get_text())
    axs[1].set_title("Hindcast - " + axs[1].title.get_text())

    # Part 2
    fig, axs = plt.subplots(2)
    lon_interv_2 = [-90, -86]
    lat_interv_2 = [23, 27]
    sel_dict = {"lon": slice(*lon_interv_2), "lat": slice(*lat_interv_2)}
    fc_frame = fc_frame.sel(**sel_dict)
    data_hc_one_time = data_hc_one_time.sel(**sel_dict)
    vmin = min(np.min(fc_frame['magnitude'].min()).item(), np.min(data_hc_one_time['magnitude'].min()).item())
    vmax = max(np.max(fc_frame['magnitude'].max()).item(), np.max(data_hc_one_time['magnitude'].max()).item())

    print("plot fc")
    # fc.plot_data_at_time_over_area(start, lon_interv, lat_interv, ax=axs[0])
    fc.plot_data_from_xarray(0, fc_frame, ax=axs[0], vmin=vmin, vmax=vmax)
    print("plot hc")
    hc.plot_data_from_xarray(0, data_hc_one_time, ax=axs[1], colorbar=False, vmin=vmin, vmax=vmax,
                             fill_space_for_cbar=False)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[0].set_title("Forecast - " + axs[0].title.get_text())
    axs[1].set_title("Hindcast - " + axs[1].title.get_text())

print("over :)")
