# import requests
#
# TOKEN = "aaf99bf8a75342d991ad9b750e2107"
# query = {"token": TOKEN, "limit": 50, "spotterId": "SPOT-0222"}
# api_endpoint = "https://api.sofarocean.com/api/wave-data"
# response = requests.get(api_endpoint, params=query)
# print(response.json())

import glob
import os

import pandas as pd

from ocean_navigation_simulator.generative_error_model.utils import (
    convert_degree_to_km,
)


def main(sofar_dir: str, output_dir: str):
    """Takes raw files from sofar and"""

    print(sofar_dir, output_dir)

    sofar_files = sorted(glob.glob(os.path.join(sofar_dir, "*.json")))

    index_file = pd.DataFrame(
        columns=[
            "file_name",
            "geospatial_lat_min",
            "geospatial_lat_max",
            "geospatial_lon_min",
            "geospatial_lon_max",
            "time_coverage_start",
            "time_coverage_end",
        ]
    )

    for file in sofar_files:
        data = pd.read_json(file)

        # rename columns to match copernicus
        data = data.rename(
            columns={
                "longitude": "lon",
                "latitude": "lat",
                "timestamp": "time",
                "spotterId": "buoy",
            }
        )

        # convert from degrees to km
        data["lon_km"], data["lat_km"] = convert_degree_to_km(data["lon"], data["lat"])

        # get difference in space and time between consecutive measurements
        data["lat_km_diff"] = data["lat_km"].diff()
        data["lon_km_diff"] = data["lon_km"].diff()
        data["time_diff"] = data["time"].diff()
        data = data.dropna()

        # delete columns where delta time is too large and finite difference makes no sense
        data = data[data["time_diff"].dt.components["hours"] <= 1]

        # convert time to seconds
        data["time_diff_secs"] = data["time_diff"].dt.seconds

        data["u"] = data["lon_km_diff"] * 1000 / data["time_diff_secs"]
        data["v"] = data["lat_km_diff"] * 1000 / data["time_diff_secs"]

        data = data.drop(
            ["lon_km", "lat_km", "lat_km_diff", "lon_km_diff", "time_diff", "time_diff_secs"],
            axis=1,
        )

        # change time format to match Copernicus
        data["time"] = data["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        file_path = os.path.join(output_dir, "".join(file.split("/")[-1])).split(".")[0] + ".csv"
        print(f"Saved: {file_path}.")
        data.to_csv(file_path, index=False)

        # add file meta data to index file
        index_file.loc[len(index_file.index)] = [
            file_path.split("/")[-1],
            data["lat"].min(),
            data["lat"].max(),
            data["lon"].min(),
            data["lon"].max(),
            data["time"].min(),
            data["time"].max(),
        ]

    index_file.to_csv(os.path.join(output_dir, "index.csv"), index=False)


if __name__ == "__main__":
    sofar_data = "../../data/drifter_data/sofar/sofar_data"
    output_dir = "../../../data/drifter_data/sofar/sofar_data_pre_processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(sofar_data, output_dir)
