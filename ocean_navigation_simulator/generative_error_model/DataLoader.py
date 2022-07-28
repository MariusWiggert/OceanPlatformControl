# Notes for implementation:
# need to replace double underscore in csv file name
# to forward slash to recover the time str part of name

from torch.utils.data import Dataset
import yaml

class CurrentErrorDataset(Dataset):
    def __init__(self, yaml_file_config):
        pass