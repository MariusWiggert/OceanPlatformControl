from ocean_navigation_simulator.generative_error_model.BuoyData import BuoyDataCopernicus
from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.generative_error_model.utils import get_path_to_project

import pandas as pd
import os
import yaml
from datetime import timedelta, datetime


class DatasetWriter:
    """Uses BuoyData and OceanCurrentField to write data to file which is then read by DataLoader.
    Needed to speed up training."""

    def __init__(self, yaml_file_config: str, input_dir: str, output_dir: str):
        with open(yaml_file_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.data_dir = os.path.join(get_path_to_project(os.getcwd()), self.config["data_dir"])
        # where forecasts are coming from
        input_path = os.path.join(self.data_dir, "forecasts/", input_dir)
        if input_path != "/":
            input_path += "/"
        self.config["local_forecast"]["source_settings"]["folder"] = input_path
        print(f"Loading data from: {input_path}")
        # where dataset is saved
        self.output_path = os.path.join(self.data_dir, "dataset_forecast_error", output_dir)
        print(f'Writing data to: {self.output_path}\n')
        self.ocean_field = OceanCurrentField(self.config["sim_cache_dict"], self.config["local_forecast"])
        self.files_dicts = self.ocean_field.forecast_data_source.files_dicts

    def get_error(self, forecast_idx: int) -> pd.DataFrame:
        # get buoy data and interpolate forecast to it
        buoy_data = BuoyDataCopernicus(self.config)

        # set idx to access specific forecast in folder
        self.ocean_field.forecast_data_source.rec_file_idx = forecast_idx
        # load file of specified idx
        self.ocean_field.forecast_data_source.load_ocean_current_from_idx()
        buoy_data.interpolate_forecast(self.ocean_field)

        # compute the error
        buoy_data.data["u_error"] = buoy_data.data["u_forecast"] - buoy_data.data["u"]
        buoy_data.data["v_error"] = buoy_data.data["v_forecast"] - buoy_data.data["v"]

        return buoy_data.data

    def _build_time_string(self, forecast_idx: int) -> str:
        t_range = self.files_dicts[forecast_idx]["t_range"]
        time_string_start = (t_range[0] + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        time_string_end = (t_range[1] + timedelta(hours=-6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        time_string = time_string_start + "/" + time_string_end
        return time_string

    def write_error_csv(self, df: pd.DataFrame, file_name: str) -> None:
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        df.to_csv(os.path.join(self.output_path, file_name), index=False)

    def write_all_files(self) -> None:
        for forecast_idx in range(len(self.files_dicts)):
            file_name = f"copernicus_forecast_error_lon_{self.config['buoy_config']['copernicus']['lon_range']}"\
                        f"_lat_{self.config['buoy_config']['copernicus']['lat_range']}"\
                        f"_time_{self.config['buoy_config']['copernicus']['time_range']}.csv"
            file_name = file_name.replace("/", "__")
            time_string = self._build_time_string(forecast_idx)
            self.config["buoy_config"]["copernicus"]["time_range"] = time_string
            if os.path.exists(os.path.join(self.output_path, file_name)):
                print(f"file already exists: {file_name}")
                continue
            forecast_error = self.get_error(forecast_idx)
            self.write_error_csv(forecast_error, file_name)
            print(f"written: {file_name}")


if __name__ == "__main__":
    # run for quick testing + generating csv files
    yaml_file_config = os.path.join(get_path_to_project(os.getcwd()), "scenarios/generative_error_model/config_buoy_data.yaml")
    data_writer = DatasetWriter(yaml_file_config, "area1", "area1")
    data_writer.write_all_files()
