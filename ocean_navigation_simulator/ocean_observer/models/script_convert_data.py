import argparse
import datetime
import json
import os

import numpy as np
import torch
import yaml
from dateutil import parser as dateParser

from ocean_navigation_simulator.ocean_observer.models.CustomOceanCurrentsDataset import \
    CustomOceanCurrentsDatasetSubgrid


def collate_fn(batch):
    batch_filtered = list(filter(lambda x: x is not None, batch))
    if not len(batch_filtered):
        return None, None
    return torch.utils.data.dataloader.default_collate(batch_filtered)


def main():
    print("script to generate the data files (.npy) used as input source for the neural networks.")
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch model')
    parser.add_argument('--yaml-file-datasets', type=str, default='',
                        help='filname of the yaml file to use to download the data in the folder scenarios/neural_networks')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model-type', type=str, default='mlp')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    train_kwargs = {'batch_size': 1}
    test_kwargs = {'batch_size': 1}

    device = torch.device("cuda" if use_cuda else "cpu")

    model_type = args.model_type
    dtype = torch.float32
    transform = None
    cfgs = json.load(open(os.getcwd() + "/config/" + args.model_type + ".json", 'r'))
    cfg_dataset = cfgs.get("cfg_dataset", {})

    # Load the training and testing files
    with open(f'scenarios/neural_networks/{args.yaml_file_datasets}.yaml') as f:
        config_datasets = yaml.load(f, Loader=yaml.FullLoader)
    with open(f'scenarios/data_generation/{model_type}_data_generation.yaml') as f:
        config_data_generation = yaml.load(f, Loader=yaml.FullLoader)

    cfg_parameters_input = config_data_generation["parameters_input"]

    start_training = dateParser.parse(cfg_parameters_input["start_training"])
    duration_training = datetime.timedelta(hours=cfg_parameters_input["duration_training_in_h"])
    duration_validation = datetime.timedelta(hours=cfg_parameters_input["duration_validation_in_h"])

    array_input = cfg_parameters_input["input_tile_dims"]
    input_tile_dims = (array_input[0], array_input[1], datetime.timedelta(hours=array_input[2]))
    array_output = cfg_parameters_input["output_tile_dims"]
    output_tile_dims = (array_output[0], array_output[1], datetime.timedelta(hours=array_output[2]))
    dataset_training = CustomOceanCurrentsDatasetSubgrid(config_datasets["training"], start_training,
                                                         start_training + duration_training,
                                                         input_tile_dims, output_tile_dims, cfg_dataset, transform,
                                                         transform, dtype=dtype, )
    dataset_validation = CustomOceanCurrentsDatasetSubgrid(config_datasets["validation"],
                                                           start_training + duration_training,
                                                           start_training + duration_training + duration_validation,
                                                           input_tile_dims, output_tile_dims, cfg_dataset, transform,
                                                           transform, dtype=dtype)

    train_loader = torch.utils.data.DataLoader(dataset_training, collate_fn=collate_fn, **train_kwargs)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, collate_fn=collate_fn, **test_kwargs)

    size_file_in_byte = cfg_parameters_input["size_per_file_in_mb"] * (2 << 20)
    for loader, folder in [(train_loader, cfg_parameters_input["folder_training"]),
                           (validation_loader, cfg_parameters_input["folder_validation"])]:
        X, y = list(), list()
        index = 0
        empty_batches = list()
        for batch_idx, (data, target) in enumerate(loader):
            if (data, target) == (None, None):
                print(f"batch {batch_idx} empty. skipped!")
                empty_batches.append(batch_idx)
                continue
            data, target = data.numpy(), target.numpy()
            if batch_idx % 10000 == 0:
                print(f"{batch_idx}/{len(loader)}")
            if data.size * data.itemsize * len(X) > size_file_in_byte:
                print(f"export X and y (file {index}, batch {batch_idx}).")
                save_list_to_file(index, X, "X", folder)
                save_list_to_file(index, y, "y", folder)
                X, y = list(), list()
                index += 1
            X.append(data)
            y.append(target)

        save_list_to_file(index, X, "X", folder)
        save_list_to_file(index, y, "y", folder)
        print(f"done exporting {folder}")
        print("empty_batches: ", empty_batches[0], empty_batches[-1], empty_batches)

    print("Export over.")


def save_list_to_file(index, ls, extension_name, folder="data_exported_2") -> None:
    arr = np.concatenate(ls, axis=0)
    path = f"{folder}/export_{index}_{extension_name}.npy"
    path_parent = os.path.dirname(path)
    if not os.path.exists(path_parent):
        os.makedirs(path_parent)
    np.save(path, arr)
    print(f"created file: {path}")


if __name__ == '__main__':
    main()
