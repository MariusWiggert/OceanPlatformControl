# Notes for implementation:
# need to replace double underscore in csv file name
# to forward slash to recover the time str part of name

from torch.utils.data import Dataset
import pandas as pd
import yaml
import glob

class CurrentErrorDataset(Dataset):
    def __init__(self, yaml_file_config):
        with open(yaml_file_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.data_dir = self.config["output_dir"]
        self.data_files = list(glob.glob(os.path.join(self.data_dir, f"*.csv")))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        errors = pd.red_csv(self.data_files[idx])
        return errors