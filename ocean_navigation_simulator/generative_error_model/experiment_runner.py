from BuoyData import BuoyDataCopernicus
from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
from data_preprocessing import plot_buoy_data

import datetime
import dateutil
import matplotlib.pyplot as plt

class ExperimentRunner:
    def __init__(self):
        yaml_path = "/home/jonas/Documents/Thesis/OceanPlatformControl/scenarios/generative_error_model/config_buoy_data.yaml"
        buoy_data = BuoyDataCopernicus(yaml_path)

        print(f"Num of buoys in spatio-temporal range: {len(set(buoy_data.index_data['platform_code']))}")
        print(min(buoy_data.data["lon"]), max(buoy_data.data["lon"]))
        print(min(buoy_data.data["lat"]), max(buoy_data.data["lat"]))
        print(min(buoy_data.data["time"]), max(buoy_data.data["time"]))
        # plot_buoy_data(buoy_data.data)

        # load local hindcast/forecast
        source_dict = buoy_data.config["local_forecast"]
        sim_cache_dict = buoy_data.config["sim_cache_dict"]

        # Create the ocean Field
        ocean_field = OceanCurrentField(hindcast_source_dict=source_dict, sim_cache_dict=sim_cache_dict)

        # interpolate hindcast/forecast to buoy locations
        buoy_data.interpolate_forecast(ocean_field)
        self.data = buoy_data.data

    def plot_corr_mag_error(self):
        u_forecast = abs(self.data["u_forecast"])
        u_error = abs(self.data["u_forecast"] - self.data["u"])
        v_forecast = abs(self.data["v_forecast"])
        v_error = abs(self.data["v_forecast"] - self.data["v"])
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
        ax1.scatter(u_forecast, u_error, s=1)
        ax2.scatter(v_forecast, v_error, s=1)
        ax1.set_xlabel("u current magnitude")
        ax1.set_ylabel("u current error")
        ax2.set_xlabel("u current magnitude")
        ax2.set_ylabel("u current error")
        plt.show()

    def plot_corr_mag_error_for_buoy(self, buoy_name:str):
        u_forecast = abs(self.data[self.data["buoy"] == buoy_name]["u_forecast"])
        u_error = abs(self.data[self.data["buoy"] == buoy_name]["u_forecast"] - self.data[self.data["buoy"] == buoy_name]["u"])
        v_forecast = abs(self.data[self.data["buoy"] == buoy_name]["v_forecast"])
        v_error = abs(self.data[self.data["buoy"] == buoy_name]["v_forecast"] - self.data[self.data["buoy"] == buoy_name]["v"])
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
        ax1.scatter(u_forecast, u_error, s=1)
        ax2.scatter(v_forecast, v_error, s=1)
        ax1.set_xlabel("u current magnitude")
        ax1.set_ylabel("u current error")
        ax2.set_xlabel("v current magnitude")
        ax2.set_ylabel("v current error")
        plt.show()

    def plot_corr_mag_error_colour_coded_buoys(self):
        buoy_names = set(self.data["buoy"].to_list())
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8))
        for buoy_name in buoy_names:
            u_forecast = abs(self.data[self.data["buoy"] == buoy_name]["u_forecast"])
            u_error = abs(self.data[self.data["buoy"] == buoy_name]["u_forecast"] - self.data[self.data["buoy"] == buoy_name]["u"])
            v_forecast = abs(self.data[self.data["buoy"] == buoy_name]["v_forecast"])
            v_error = abs(self.data[self.data["buoy"] == buoy_name]["v_forecast"] - self.data[self.data["buoy"] == buoy_name]["v"])
            ax1.scatter(u_forecast, u_error, s=1)
            ax2.scatter(v_forecast, v_error, s=1)
            ax1.set_xlabel("u current magnitude")
            ax1.set_ylabel("u current error")
            ax2.set_xlabel("v current magnitude")
            ax2.set_ylabel("v current error")
        plt.show()

    def plot_corr_mag_error_per_day_colour(self):
        self.data["day"] = self.data["time"].apply(lambda x: x.day)
        days = sorted(set(self.data["day"].tolist()))
        self.data["hour"] = self.data["time"].apply(lambda x: x.hour)
        hours = sorted(set(self.data["hour"].tolist()))

        fig2, axs = plt.subplots(len(days)-2,2,figsize=(4,25), sharex=True, sharey=True)
        plt.setp(axs, xlim=(0,0.5), ylim=(0,0.5))

        for i, day in enumerate(days[1:-1]):
            for hour in hours:
                u_forecast = abs(self.data[(self.data["day"] == day) & (self.data["hour"] == hour)]["u_forecast"])
                u_error = abs(self.data[(self.data["day"] == day) & (self.data["hour"] == hour)]["u_forecast"] \
                    - self.data[(self.data["day"] == day) & (self.data["hour"] == hour)]["u"])
                v_forecast = abs(self.data[(self.data["day"] == day) & (self.data["hour"] == hour)]["v_forecast"])
                v_error = abs(self.data[(self.data["day"] == day) & (self.data["hour"] == hour)]["v_forecast"] \
                    - self.data[(self.data["day"] == day) & (self.data["hour"] == hour)]["v"])
                axs[i,0].scatter(u_forecast, u_error, s=1)
                axs[i,1].scatter(v_forecast, v_error, s=1)
                axs[i,0].set_xlabel("u magnitude")
                axs[i,0].set_ylabel("u error")
                axs[i,0].set(adjustable='box', aspect='equal')
                axs[i,1].set_xlabel("v magnitude")
                axs[i,1].set_ylabel("v error")
                axs[i,1].set(adjustable='box', aspect='equal')
        plt.show()

if __name__ == "__main__":
    # init experiment runner class
    ex_runner = ExperimentRunner()
    print(ex_runner.data)
    # plot correlation for all buoys
    ex_runner.plot_corr_mag_error_colour_coded_buoys()

