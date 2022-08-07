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


# Data loading util
import pandas as pd
import numpy as np
import os

def load_test_data() -> pd.DataFrame:
    dataset_dir = "/home/jonas/Documents/Thesis/OceanPlatformControl/data/drifter_data/dataset_forecast_error/"
    dataset_files = os.listdir(dataset_dir)

    # loop over all csv files
    for i in range(len(dataset_files)):
        if i == 0:
            df = pd.read_csv(os.path.join(dataset_dir, dataset_files[i]))
        else:
            df_temp = pd.read_csv(os.path.join(dataset_dir, dataset_files[i]))
            df = pd.concat([df, df_temp], ignore_index=True)

    # print(f"Num of rows: {df.shape[0]}")
    # print(f"Num of NaN vals: {np.isnan(df['u_error']).sum()}")
    return df

def load_single_file(file_idx: int) -> pd.DataFrame:
    data_dir = "/home/jonas/Documents/Thesis/OceanPlatformControl/data/drifter_data/dataset_forecast_error"
    file_list = os.listdir(data_dir)
    file_path = os.path.join(data_dir, file_list[file_idx])
    df = pd.read_csv(file_path)
    print(f"loaded: {file_list[file_idx]}")
    return df