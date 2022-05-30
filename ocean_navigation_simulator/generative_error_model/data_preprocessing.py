"""
functions to interpolate forecasts/hindcasts to spatio-temporal points
for buoy data
"""

from ocean_navigation_simulator.env.PlatformState import PlatformState
from ocean_navigation_simulator.env.utils import units
import pandas as pd

def interp_hindcast_xarray(df: pd.DataFrame, n: int=10) -> pd.DataFrame: 
    from tqdm import tqdm
    df["u_hind"] = 0
    df["v_hind"] = 0
    # convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])

    for i in tqdm(range(0, df.shape[0], n)):
        # hindcast_interp = ds_hind.interp(time=df.iloc[i:i+n]["time"],
        #                                 lon=df.iloc[i:i+n]["lon"],
        #                                 lat=df.iloc[i:i+n]["lat"])
        hindcast_interp = ocean_field.hindcast_data_source.DataArray.interp(time=df.iloc[i:i+n]["time"],
                                                                            lon=df.iloc[i:i+n]["lon"],
                                                                            lat=df.iloc[i:i+n]["lat"])
        # add columns to dataframe
        df["u_hind"][i:i+n] = hindcast_interp["water_u"].values.diagonal().diagonal()
        df["v_hind"][i:i+n] = hindcast_interp["water_v"].values.diagonal().diagonal()
    return df