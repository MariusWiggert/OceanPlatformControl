# write csv files with errors from one forecast to one file
# check some folder and loop over all forecast contained
# use BuoyData and OceanCurrentField class to get the error per forecast
# save to file with meaningful Name

from ocean_navigation_simulator.generative_error_model.BuoyData import BuoyDataCopernicus
from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
import pandas as pd
import os
import yaml

class DatasetWriter:
    """Uses BuoyData and OceanCurrentField to to write data to
    file which is then read by DataLoader. Needed to speed up
    training."""

    def __init__(self, yaml_file_config: str):
        # TODO: Need to figure out the location/time management in config
        with open(yaml_file_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader) 
        self.output_dir = self.config["data_writer"]["output_dir"]
        self.forecast_paths = []
        self.ocean_field = OceanCurrentField(self.config["sim_cache_dict"], self.config["local_forecast"])
        self.files_dicts = self.ocean_field.forecast_data_source.files_dicts

    def get_error(self, forecast_idx: int) -> pd.DataFrame:
        # get buoy data and interpolate forecast to it
        buoy_data = BuoyDataCopernicus(self.config)

        # TODO: Specify time for buoys to be within forecast time
        time_string = self._build_time_string(forecast_idx)
        self.config["buoy_config"]["copernicus"]["time_range"] = time_string

        # set idx to access specific forecast in folder
        self.ocean_field.forecast_data_source.rec_file_idx = forecast_idx
        buoy_data.interpolate_forecast(self.ocean_field)

        # compute the error
        buoy_data.data["u_error"] = buoy_data.data["u_forecast"] - buoy_data.data["u"]
        buoy_data.data["v_error"] = buoy_data.data["v_forecast"] - buoy_data.data["v"]

        return buoy_data.data

    def _build_time_string(self, forecast_idx: int) -> str:
        t_range = self.files_dicts[forecast_idx]["t_range"]
        "2022-04-24T13:00:00Z/2022-05-03T13:00:00Z"
        time_string_start = t_range[0].strftime("%Y-%m-%dT%H:%M:%SZ")
        time_string_end = t_range[1].strftime("%Y-%m-%dT%H:%M:%SZ")
        time_string = time_string_start + "/" + time_string_end
        return time_string

    def write_error_csv(self, df: pd.DataFrame, file_name: str) -> None:
        # check if dir exists
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        df.to_csv(os.path.join(self.output_dir, file_name), index=False)

    def write_all_files(self) -> None:
        # write all files
        for forecast_idx in range(len(self.files_dicts)):
            forecast_error = self.get_error(forecast_idx)
            file_name = f"copernicus_forecast_error_lon_{self.config['buoy_config']['copernicus']['lon_range']}"\
                        f"_lat_{self.config['buoy_config']['copernicus']['lat_range']}"\
                        f"_time_{self.config['buoy_config']['copernicus']['time_range']}.csv"
            file_name = file_name.replace("/", "__")
            self.write_error_csv(forecast_error, file_name)
            print(f"written: {file_name}")


if __name__ == "__main__":
    # run for quick testing
    yaml_file_config = "/home/jonas/Documents/Thesis/OceanPlatformControl/scenarios/generative_error_model/config_buoy_data.yaml"
    datawriter = DatasetWriter(yaml_file_config)
    datawriter.write_all_files()