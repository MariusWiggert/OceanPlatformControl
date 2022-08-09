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


from enum import Enum, auto
import pandas as pd
import numpy as np
import os


class DatasetName(Enum):
    AREA1 = auto()
    AREA3 = auto()

dataset_map = {DatasetName.AREA1: "area1", DatasetName.AREA3: "area3"}


def load_dataset(dataset_name: DatasetName) -> pd.DataFrame:
    dataset_root = "/home/jonas/Documents/Thesis/OceanPlatformControl/data/drifter_data/dataset_forecast_error/"
    dataset_path = os.path.join(dataset_root, dataset_map[dataset_name])
    dataset_files = os.listdir(dataset_path)

    for i in range(len(dataset_files)):
        if i == 0:
            df = pd.read_csv(os.path.join(dataset_path, dataset_files[i]))
        else:
            df_temp = pd.read_csv(os.path.join(dataset_path, dataset_files[i]))
            df = pd.concat([df, df_temp], ignore_index=True)
    print(f"Loaded {dataset_name.name} dataset.")
    return df


def load_single_file(dataset_name: DatasetName, file_idx: int) -> pd.DataFrame:
    dataset_root = "/home/jonas/Documents/Thesis/OceanPlatformControl/data/drifter_data/dataset_forecast_error"
    dataset_path = os.path.join(dataset_root, dataset_map[dataset_name])
    file_list = os.listdir(dataset_path)
    file_path = os.path.join(dataset_path, file_list[file_idx])
    df = pd.read_csv(file_path)
    print(f"loaded: {file_list[file_idx]}")
    return df


if __name__ == "__main__":
    dataset_name = DatasetName.AREA1
    data = load_single_file(dataset_name, file_idx=0)
    print("\nBuoy Meta Data:")
    print(f"Min time: {data['time'].min()}, max time: {data['time'].max()}")
    print(f"Min lon: {data['lon'].min()}, max lon: {data['lon'].max()}")
    print(f"Min lat: {data['lat'].min()}, max lat: {data['lat'].max()}\n")