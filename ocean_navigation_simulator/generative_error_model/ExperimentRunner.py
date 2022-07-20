from BuoyData import BuoyDataCopernicus
from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField

class ExperimentRunner:
    def __init__(self):
        # get buoy data and interpolate forecast to it
        yaml_path = "/home/jonas/Documents/Thesis/OceanPlatformControl/scenarios/generative_error_model/config_buoy_data.yaml"
        buoy_data = BuoyDataCopernicus(yaml_path)

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
    