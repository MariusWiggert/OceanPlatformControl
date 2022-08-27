# # Notes for implementation:
# # need to replace double underscore in csv file name
# # to forward slash to recover the time str part of name

# from torch.utils.data import Dataset
# import pandas as pd
# import yaml
# import glob

# class CurrentErrorDataset(Dataset):
#     def __init__(self, yaml_file_config):
#         with open(yaml_file_config) as f:
#             self.config = yaml.load(f, Loader=yaml.FullLoader)
#         self.data_dir = self.config["output_dir"]
#         self.data_files = list(glob.glob(os.path.join(self.data_dir, f"*.csv")))

#     def __len__(self):
#         return len(self.data_files)

#     def __getitem__(self, idx):
#         errors = pd.red_csv(self.data_files[idx])
#         return errors

from ocean_navigation_simulator.generative_error_model.utils import get_path_to_project, load_config
import pandas as pd
import os
from typing import List, Dict
import datetime


class Dataset:
    """This class handles the data loading for the Variogram computation and the ExperimentRunner.
    """
    def __init__(self, dataset_name: str, config: Dict):
        self.config = config

        # check if dataset exists
        root = get_path_to_project(os.getcwd())
        self.dataset_type = config["dataset_type"]
        self.dataset_dir = os.path.join(root, self.config["data_dir"], f"dataset_{self.dataset_type}_error")
        self.datasets = os.listdir(self.dataset_dir)
        if dataset_name not in self.datasets:
            raise ValueError(f"Specified dataset {dataset_name} does not exist!")

        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(self.dataset_dir, dataset_name)
        self.meta_data = self.build_meta_data_list()

    def load_dataset(self, overlap: bool=True, verbose: bool=True) -> pd.DataFrame:
        """Loads entire dataset specified.
        """
        dataset_files = os.listdir(self.dataset_path)
        df = pd.read_csv(os.path.join(self.dataset_path, dataset_files[0]))
        df = pd.DataFrame(columns=df.columns)

        for i in range(len(dataset_files)):
            df_temp = pd.read_csv(os.path.join(self.dataset_path, dataset_files[i]))
            if overlap is False:
                times = sorted(list(set(df_temp["time"])))[:24]
                df_temp = df_temp[df_temp["time"].isin(times)]
            df = pd.concat([df, df_temp], ignore_index=True)
        print(f"Loaded {self.dataset_name} from {self.dataset_type}s.")
        if verbose:
            self.print_df_meta_data(df)
        return df

    def load_single_file(self, file_idx: int = 0) -> pd.DataFrame:
        """Loads any file of specified dataset.
        """
        file_list = os.listdir(self.dataset_path)
        file_path = os.path.join(self.dataset_path, file_list[file_idx])
        df = pd.read_csv(file_path)
        print(f"Loaded: {file_list[file_idx]}")
        self.print_df_meta_data(df)
        return df

    def load_file(self, file_name: str) -> pd.DataFrame:
        """Loads a specific file of name file_name.
        """
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
            temp["lon_range"] = [float(file_split[4].strip("[").strip("]").split(",")[0]),
                                 float(file_split[4].strip("[").strip("]").split(",")[1])]
            temp["lat_range"] = [float(file_split[6].strip("[").strip("]").split(",")[0]),
                                 float(file_split[6].strip("[").strip("]").split(",")[1])]
            temp["t_range"] = [file_split[8], file_split[10].split(".")[0]]
            list_of_dicts.append(temp)
        list_of_dicts.sort(key=lambda dictionary: dictionary['t_range'][0])
        return list_of_dicts

    def get_specific_data(self, lon_range, lat_range, t_range: List[datetime.datetime]) -> pd.DataFrame:
        """Gets the most up to date date for a specific range.
        """
        df = self.load_dataset(overlap=False, verbose=False)
        # TODO: logic here is slightly flawed. Example, only one file -> only one day of points
        df = df[(df["time"] >= t_range[0].strftime("%Y-%m-%d %H:%M:%S")) &
                (df["time"] <= t_range[1].strftime("%Y-%m-%d %H:%M:%S"))]
        df = df[(df["lon"] >= lon_range[0]+1) & (df["lon"] <= lon_range[1]-1)]
        df = df[(df["lat"] >= lat_range[0]+1) & (df["lat"] <= lat_range[1]-1)]
        return df

    def print_df_meta_data(self, data: pd.DataFrame):
        """Convenience method for printing meta data after loading data.
        """
        print("\nBuoy Meta Data:")
        print(f"    Min time: {data['time'].min()}, max time: {data['time'].max()}")
        print(f"    Min lon: {data['lon'].min()}, max lon: {data['lon'].max()}")
        print(f"    Min lat: {data['lat'].min()}, max lat: {data['lat'].max()}")
        print(f"    Number or rows: {len(data)}.\n")


if __name__ == "__main__":
    config = load_config("config_buoy_data.yaml")
    problems = config["experiment_runner"]["problems"][0]["data_ranges"]
    dataset = Dataset("area1", config)
    import dateutil
    start = dateutil.parser.isoparse(problems["t_range"].split("/")[0])
    end = dateutil.parser.isoparse(problems["t_range"].split("/")[1])
    data = dataset.get_specific_data(lon_range=problems["lon_range"],
                                     lat_range=problems["lat_range"],
                                     t_range=[start, end])
    print(data)
