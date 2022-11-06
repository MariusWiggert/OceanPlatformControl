#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
import os

import numpy as np
from npy_append_array import NpyAppendArray


# In[7]:

# File used to sample the output of the missions such that it can be used as NN input without having to much
# correlation between the elements


def remove_borders_GP_predictions_lon_lat(
        x,
        radius_to_keep,
        channels_to_0=[0, 1, 2, 3],
        channels_to_dest=[6, 7],
        dest_channels=[4, 5],
):
    middle = x.shape[-1] // 2
    radius_to_remove = (x.shape[-1] - radius_to_keep * 2) / 2
    assert middle == int(middle) and radius_to_remove == int(radius_to_remove)
    middle, radius_to_remove = int(middle), int(radius_to_remove)
    x[:, channels_to_0, :, :radius_to_remove, :] = 0
    x[:, channels_to_0, :, -radius_to_remove:, :] = 0
    x[:, channels_to_0, :, :, :radius_to_remove] = 0
    x[:, channels_to_0, :, :, -radius_to_remove:] = 0
    x[:, channels_to_dest, :, :radius_to_remove, :] = x[:, dest_channels, :, :radius_to_remove, :]
    x[:, channels_to_dest, :, -radius_to_remove:, :] = x[:, dest_channels, :, -radius_to_remove:, :]
    x[:, channels_to_dest, :, :, :radius_to_remove] = x[:, dest_channels, :, :, :radius_to_remove]
    x[:, channels_to_dest, :, :, -radius_to_remove:] = x[:, dest_channels, :, :, -radius_to_remove:]
    return x


def main():
    print("start main")
    parser = argparse.ArgumentParser(description="yaml config file path")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--folder-output", type=str)
    parser.add_argument("--filename", type=str)
    parser.add_argument("--index", type=int)
    parser.add_argument("--training", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    i = args.index
    if args.training:
        folder = args.folder
        output_folder = args.folder_output
    else:
        # todo: to fix
        print(f"step: {i}")
        folder = f"/home/killian2k/seaweed/OceanPlatformControl/data_NN_DA/validation/copy_{i}/"
        output_folder = f"/home/killian2k/seaweed/OceanPlatformControl/data_NN_DA/GP_all_files_validation/copy_{i}/"
    x = np.load(f"{folder}{args.filename}_x.npy", mmap_mode="r")
    out_x = f"{output_folder}{args.filename}_{i}_x.npy"
    y = np.load(f"{folder}{args.filename}_y.npy", mmap_mode="r")
    out_y = f"{output_folder}{args.filename}_{i}_y.npy"

    # error = np.load("/datadrive/files_copy_1/error.csv", mmap_mode='r')
    # measurement = np.load("/datadrive/files_copy_1/measurement.csv", mmap_mode='r')
    print(x[0, :, 0, 0])
    print(x.shape, y.shape)
    print(x.reshape((-1, 12, 8, 25, 25))[0, 0, :, 0, 0])
    # We have 8 and 2 channels for x and y respectively and 1323 problems saved. Dim2 = 12 hours * 96 days, 24,24 for lon and lat
    # x_reshaped, y_reshaped = x.reshape((-1, 8, 96, 12, 24, 24)), y.reshape((-1, 2, 96, 12, 24, 24))
    # x_reshaped, y_reshaped = x.reshape((-1, 8, 12, 25, 25)), y.reshape((-1, 2, 12, 25, 25))
    x_reshaped_2, y_reshaped_2 = x.reshape((-1, 12, 8, 25, 25)), y.reshape((-1, 12, 2, 25, 25))
    # print(x_reshaped == x_reshaped_2, y_reshaped == y_reshaped_2)
    # x_reshaped = x_reshaped[:len(y_reshaped)]
    # x_reshaped = np.swapaxes(x_reshaped, 1, 2)
    # y_reshaped = np.swapaxes(y_reshaped, 1, 2)
    print("shapes: ", x_reshaped_2.shape, y_reshaped_2.shape)
    print(
        "zero:",
        x_reshaped_2[0, 0, :, 0, 0],
        x_reshaped_2[0, 0, 0, 0, 0] - x_reshaped_2[0, 0, 4, 0, 0],
        -x_reshaped_2[0, 0, 6, 0, 0],
    )
    # print("shapes: ", x_reshaped.shape, y_reshaped.shape)
    # In[8]:

    x, y = [], []
    isExist = os.path.exists(output_folder)
    if not isExist:
        # Create a new directory because it does not exist
        print(f"create path: {output_folder}")
        os.makedirs(output_folder)

    num_samples_per_period = 3
    num_hours_between_period = 12
    num_hours_total = 96
    i = 0
    with NpyAppendArray(out_x) as npaa_x:
        with NpyAppendArray(out_y) as npaa_y:
            while i < len(x_reshaped_2):
                # for i, (problem, y_problem) in enumerate(zip(x_reshaped, y_reshaped)):
                if i % 10 == 0:
                    print(f"{i}/{len(x_reshaped_2)}")
                samples = []
                for k in range(0, num_hours_total, num_hours_between_period):
                    samples += list(
                        k
                        + np.random.choice(
                            num_hours_between_period,
                            size=num_samples_per_period,
                            replace=False,
                        )
                    )
                samples = np.array(samples)
                # print(problem[:,j, c])ws
                # print(y_reshaped[i,:,j,c])
                # x.append(problem[:,samples])
                # y.append(y_reshaped[i,:,samples])
                x = x_reshaped_2[i + samples]
                y = y_reshaped_2[i + samples]
                # x = remove_borders_GP_predictions_lon_lat(x)
                assert x.shape[0] == y.shape[0]
                npaa_x.append(np.ascontiguousarray(np.ascontiguousarray(x).swapaxes(1, 2)))
                npaa_y.append(np.ascontiguousarray(np.ascontiguousarray(y).swapaxes(1, 2)))
                i += num_hours_total

    # In[9]:

    # x = np.concatenate(x).reshape(-1, 8, num_samples_in_total, *x_reshaped_2.shape[-3:])

    # In[10]:

    # y = np.concatenate(y).reshape(-1, 2, num_samples_in_total, *x_reshaped_.shape[-3:])

    # In[12]:

    # x.shape, y.shape


if __name__ == "__main__":
    main()
