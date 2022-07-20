from BuoyData import BuoyDataCopernicus
from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField

import yaml

class ExperimentRunner:
    def __init__(self):
        # harcoded config file for now
        yaml_file_config = "/home/jonas/Documents/Thesis/OceanPlatformControl/scenarios/generative_error_model/config_buoy_data.yaml"

        # read in yaml config file
        with open(yaml_file_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        # get buoy data and interpolate forecast to it
        buoy_data = BuoyDataCopernicus(self.config)

        print(f"Num of buoys in spatio-temporal range: {len(set(buoy_data.index_data['platform_code']))}")
        # print(min(buoy_data.data["lon"]), max(buoy_data.data["lon"]))
        # print(min(buoy_data.data["lat"]), max(buoy_data.data["lat"]))
        # print(min(buoy_data.data["time"]), max(buoy_data.data["time"]))

        # load local hindcast/forecast
        source_dict = buoy_data.config["local_forecast"]
        sim_cache_dict = buoy_data.config["sim_cache_dict"]

        # Create the ocean Field
        ocean_field = OceanCurrentField(hindcast_source_dict=source_dict, sim_cache_dict=sim_cache_dict)

        # interpolate hindcast/forecast to buoy locations
        buoy_data.interpolate_forecast(ocean_field)
        self.data = buoy_data.data

        # in future read in further configs like model and hyper params etc

    def visualize(self):
        pass

    def sample_from_model(self):
        pass

    def calulate_metrics(self):
        pass

    def plot_metrics(self):
        pass
    