import argparse
import datetime
import os

import numpy as np
import torch
import yaml
from dateutil import parser as dateParser

from ocean_navigation_simulator.ocean_observer.Other.DotDict import DotDict
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
    # parser.add_argument('--yaml-file-datasets', type=str, default='',
    #                     help='filname of the yaml file to use to download the data in the folder scenarios/neural_networks')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--model-type', type=str, default='mlp')
    # args = parser.parse_args()

    parser.add_argument('--file-configs', type=str, help='name file config to run (without the extension)')
    all_cfgs = yaml.load(open(parser.parse_args().file_configs + ".yaml", 'r'),
                         Loader=yaml.FullLoader)
    args = all_cfgs.get("arguments_script_convert_data", {})
    args.setdefault("yaml_file_datasets", "")
    args.setdefault("model_type", "mlp")
    args = DotDict(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    train_kwargs = {'batch_size': 1}
    test_kwargs = {'batch_size': 1}

    dtype = torch.float32
    transform = None
    cfgs = yaml.load(open(parser.parse_args().file_configs + ".yaml", 'r'), Loader=yaml.FullLoader)
    cfg_model = cfgs.get("model", {})
    cfg_dataset = cfg_model.get("cfg_dataset", {})

    # Load the training and testing files
    # with open(f'scenarios/neural_networks/{args.yaml_file_datasets}.yaml') as f:
    #     config_datasets = yaml.load(f, Loader=yaml.FullLoader)
    config_datasets = cfgs.get("config_neural_net", {})
    # with open(f'scenarios/data_generation/{model_type}_data_generation.yaml') as f:
    #     config_data_generation = yaml.load(f, Loader=yaml.FullLoader)
    config_data_generation = cfgs.get("data_generation", {})

    cfg_parameters_input = config_data_generation["parameters_input"]

    start_training = dateParser.parse(cfg_parameters_input["start_training"])
    duration_training = datetime.timedelta(hours=cfg_parameters_input["duration_training_in_h"])
    # start_validation = dateParser.parse(cfg_parameters_input["start_validation"])
    # duration_validation = datetime.timedelta(hours=cfg_parameters_input["duration_validation_in_h"])

    array_input = cfg_parameters_input["input_tile_dims"]
    input_tile_dims = (array_input[0], array_input[1], datetime.timedelta(hours=array_input[2]))
    array_output = cfg_parameters_input["output_tile_dims"]
    output_tile_dims = (array_output[0], array_output[1], datetime.timedelta(hours=array_output[2]))
    dataset_training = CustomOceanCurrentsDatasetSubgrid(config_datasets["training"], start_training,
                                                         start_training + duration_training,
                                                         input_tile_dims, output_tile_dims, cfg_dataset, dtype=dtype)
    # dataset_validation = CustomOceanCurrentsDatasetSubgrid(config_datasets["validation"],
    #                                                        start_validation,
    #                                                        start_validation + duration_validation,
    #                                                        input_tile_dims, output_tile_dims, cfg_dataset, dtype=dtype)

    train_loader = torch.utils.data.DataLoader(dataset_training, collate_fn=collate_fn, **train_kwargs)
    # validation_loader = torch.utils.data.DataLoader(dataset_validation, collate_fn=collate_fn, **test_kwargs)

    size_file_in_byte = cfg_parameters_input["size_per_file_in_mb"] * (2 << 20)
    for loader, folder in [(train_loader, cfg_parameters_input["folder_training"]),
                           ]:  # (validation_loader, cfg_parameters_input["folder_validation"])]:
        print(f"starting adding files to {folder}.")
        X, y = list(), list()
        index = 0
        # empty_batches = list()
        total_nans = 0
        for batch_idx, (data, target) in enumerate(loader):
            if not batch_idx % 5000:
                print(f"progress: {batch_idx}/{len(loader)}")
            if (data, target) == (None, None):
                # print(f"batch {batch_idx} empty. skipped!")
                total_nans += 1
                # empty_batches.append(batch_idx)
                continue
            data, target = data.numpy(), target.numpy()
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
        # print("empty_batches: ", empty_batches[0], empty_batches[-1], empty_batches)
        print(f"ratio of nans: {total_nans / len(loader)}")

    print("Export over.")


def save_list_to_file(index, ls, extension_name, folder) -> None:
    arr = np.concatenate(ls, axis=0)
    path = f"{folder}export_{index}_{extension_name}.npy"
    path_parent = os.path.dirname(path)
    if not os.path.exists(path_parent):
        os.makedirs(path_parent)
    np.save(path, arr)
    print(f"created file: {path}")


if __name__ == '__main__':
    main()
