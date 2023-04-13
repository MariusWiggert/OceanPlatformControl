from ocean_navigation_simulator.generative_error_model.utils import get_path_to_project, load_config
import pandas as pd
import os
from typing import List
import datetime


class Dataset:
    """This class handles the data loading for the Variogram computation and the ExperimentRunner."""

    def __init__(self, data_dir: str, dataset_type: str, dataset_name: str):
        # TODO: change this convoluted way of deciding what data to load.

        # check if dataset exists
        root = get_path_to_project(os.getcwd())
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.dataset_dir = os.path.join(root, data_dir, f"dataset_{dataset_type}_error")
        self.datasets = os.listdir(self.dataset_dir)
        if dataset_name not in self.datasets:
            raise ValueError(f"Specified dataset {dataset_name} does not exist!")

        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(self.dataset_dir, dataset_name)
        # TODO: find better way to get meta-data -> e.g. read in files and check extremal values.
        # self.meta_data = self.build_meta_data_list()

    def load_dataset(self, overlap: bool = True, verbose: bool = False) -> pd.DataFrame:
        """Loads entire dataset specified."""
        dataset_files = os.listdir(self.dataset_path)
        df = pd.read_csv(os.path.join(self.dataset_path, dataset_files[0]))
        df = pd.DataFrame(columns=df.columns)

        for i in range(len(dataset_files)):
            df_temp = pd.read_csv(os.path.join(self.dataset_path, dataset_files[i]))
            if overlap is False:
                times = sorted(list(set(df_temp["time"])))[:24]
                df_temp = df_temp[df_temp["time"].isin(times)]
            df = pd.concat([df, df_temp], ignore_index=True)
        print(f"Loaded {self.dataset_name} from {self.dataset_type}.")
        if verbose:
            self.print_df_meta_data(df)
        return df

    def load_single_file(self, file_idx: int = 0) -> pd.DataFrame:
        """Loads any file of specified dataset."""
        file_list = os.listdir(self.dataset_path)
        file_path = os.path.join(self.dataset_path, file_list[file_idx])
        df = pd.read_csv(file_path)
        print(f"Loaded: {file_list[file_idx]}")
        self.print_df_meta_data(df)
        return df

    def load_file(self, file_name: str) -> pd.DataFrame:
        """Loads a specific file of name file_name."""
        if file_name not in os.listdir(self.dataset_path):
            raise FileExistsError(f"Specified file '{file_name}' does not exist!")
        df = pd.read_csv(os.path.join(self.dataset_path, file_name))
        return df

    def build_meta_data_list(self):
        list_of_dicts = []
        files = os.listdir(self.dataset_path)
        for file in files:
            file_split = file.split("_")
            temp = dict()
            temp["file_name"] = file
            temp["lon_range"] = [
                float(file_split[4].strip("[").strip("]").split(",")[0]),
                float(file_split[4].strip("[").strip("]").split(",")[1]),
            ]
            temp["lat_range"] = [
                float(file_split[6].strip("[").strip("]").split(",")[0]),
                float(file_split[6].strip("[").strip("]").split(",")[1]),
            ]
            temp["t_range"] = [file_split[8], file_split[10].split(".")[0]]
            list_of_dicts.append(temp)
        list_of_dicts.sort(key=lambda dictionary: dictionary["t_range"][0])
        return list_of_dicts

    def get_recent_data_in_range(
        self, lon_range, lat_range, t_range: List[datetime.datetime]
    ) -> pd.DataFrame:
        """Gets the most up-to-date date for a specific range."""
        # gets first day of each file
        df = self.load_dataset(overlap=False, verbose=False)

        # need to get all days of last file (as they are the most up-to-date available)
        last_file_name = self.meta_data[-1]["file_name"]
        last_file = pd.read_csv(os.path.join(self.dataset_path, last_file_name))
        times_of_interest = sorted(list(set(last_file["time"])))[24:]
        last_file = last_file[last_file["time"].isin(times_of_interest)]
        df = pd.concat([df, last_file], ignore_index=True)

        # filter for specific ranges
        df = df[
            (df["time"] >= t_range[0].strftime("%Y-%m-%d %H:%M:%S"))
            & (df["time"] <= t_range[1].strftime("%Y-%m-%d %H:%M:%S"))
        ]
        df = df[(df["lon"] >= lon_range[0]) & (df["lon"] <= lon_range[1])]
        df = df[(df["lat"] >= lat_range[0]) & (df["lat"] <= lat_range[1])]
        return df

    def print_df_meta_data(self, data: pd.DataFrame):
        """Convenience method for printing meta data after loading data."""
        print("\nBuoy Meta Data:")
        print(f"    Min time: {data['time'].min()}, max time: {data['time'].max()}")
        print(f"    Min lon: {data['lon'].min()}, max lon: {data['lon'].max()}")
        print(f"    Min lat: {data['lat'].min()}, max lat: {data['lat'].max()}")
        print(f"    Number or rows: {len(data)}.\n")


if __name__ == "__main__":
    config = load_config("config_buoy_data.yaml")
    data_dir = "data/drifter_data"
    dataset = Dataset(data_dir, "synthetic", "area1")
    data = dataset.load_file(
        "synthetic_data_error_lon_[-140,-120]_lat_[20,30]_time_2022-04-21T13:00:00Z__2022-04-30T13:00:00Z.csv"
    )
    print(data)
