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

from ocean_navigation_simulator.generative_error_model.utils import load_config
import pandas as pd
import os


class Dataset:

    def __init__(self, dataset_name: str):
        self.config = load_config()
        self.dataset_dir = os.path.join(self.config["data_dir"], "dataset_forecast_error/")
        self.datasets = os.listdir(self.dataset_dir)

        if dataset_name not in self.datasets:
            raise ValueError(f"Specified dataset {dataset_name} does not exist!")
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(self.dataset_dir, dataset_name)

    def load_dataset(self, overlap: bool=True) -> pd.DataFrame:

        dataset_files = os.listdir(self.dataset_path)

        for i in range(len(dataset_files)):
            if i == 0:
                df = pd.read_csv(os.path.join(self.dataset_path, dataset_files[i]))
                if overlap is False:
                    times = sorted(list(set(df["time"])))[:24]
                    df = df[df["time"].isin(times)]
            else:
                df_temp = pd.read_csv(os.path.join(self.dataset_path, dataset_files[i]))
                if overlap is False:
                    times = sorted(list(set(df_temp["time"])))[:24]
                    df_temp = df_temp[df_temp["time"].isin(times)]
                df = pd.concat([df, df_temp], ignore_index=True)
        print(f"Loaded {self.dataset_name} dataset.")
        self.print_df_meta_data(df)
        return df

    def load_single_file(self, file_idx: int = 0) -> pd.DataFrame:
        file_list = os.listdir(self.dataset_path)
        file_path = os.path.join(self.dataset_path, file_list[file_idx])
        df = pd.read_csv(file_path)
        print(f"Loaded: {file_list[file_idx]}")
        self.print_df_meta_data(df)
        return df

    def print_df_meta_data(self, data: pd.DataFrame):
        print("\nBuoy Meta Data:")
        print(f"    Min time: {data['time'].min()}, max time: {data['time'].max()}")
        print(f"    Min lon: {data['lon'].min()}, max lon: {data['lon'].max()}")
        print(f"    Min lat: {data['lat'].min()}, max lat: {data['lat'].max()}")
        print(f"    Number or rows: {len(data)}.\n")


if __name__ == "__main__":
    dataset = Dataset("area1_small")
    data = dataset.load_single_file(file_idx=0)
