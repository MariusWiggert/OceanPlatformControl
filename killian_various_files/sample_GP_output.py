#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

import numpy as np
from npy_append_array import NpyAppendArray


# In[7]:

# dims: problem, #channels, time, lag, lon, lat
# sample 3 samples per day and we have 4 days -> 96

def remove_borders_GP_predictions_lon_lat(x, radius_to_keep, channels_to_0=[0, 1, 2, 3], channels_to_dest=[6, 7],
                                          dest_channels=[4, 5]):
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
    validation = False
    i = 0
    print(f"step: {i}")
    if not validation:
        folder = "/home/killian2k/seaweed/OceanPlatformControl/data_NN_DA/export/"
        output_folder = f"/home/killian2k/seaweed/OceanPlatformControl/data_NN_DA/GP_all_files/test_{i}/"
    else:
        folder = f"/home/killian2k/seaweed/OceanPlatformControl/data_NN_DA/validation/copy_{i}/"
        output_folder = f"/home/killian2k/seaweed/OceanPlatformControl/data_NN_DA/GP_all_files_validation/copy_{i}/"
    x = np.load(f"{folder}data_x.npy", mmap_mode='r')
    out_x = f"{output_folder}out_x.npy"
    y = np.load(f"{folder}data_y.npy", mmap_mode='r')
    out_y = f"{output_folder}out_y.npy"

    # error = np.load("/datadrive/files_copy_1/error.csv", mmap_mode='r')
    # measurement = np.load("/datadrive/files_copy_1/measurement.csv", mmap_mode='r')
    print(x.shape, y.shape)
    # We have 8 and 2 channels for x and y respectively and 1323 problems saved. Dim2 = 12 hours * 96 days, 24,24 for lon and lat
    x_reshaped, y_reshaped = x.reshape((-1, 8, 96, 12, 24, 24)), y.reshape((-1, 2, 96, 12, 24, 24))
    x_reshaped = x_reshaped[:len(y_reshaped)]
    x_reshaped = np.swapaxes(x_reshaped, 1, 2)
    y_reshaped = np.swapaxes(y_reshaped, 1, 2)
    x_reshaped.shape, y_reshaped.shape

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
    num_samples_in_total = num_samples_per_period * num_hours_total // num_hours_between_period
    with NpyAppendArray(out_x) as npaa_x:
        with NpyAppendArray(out_y) as npaa_y:
            for i, (problem, y_problem) in enumerate(zip(x_reshaped, y_reshaped)):
                if i % 10 == 0:
                    print(f"{i}/{len(x_reshaped)}")
                samples = []
                for k in range(0, num_hours_total, num_hours_between_period):
                    samples += list(
                        k + np.random.choice(num_hours_between_period, size=num_samples_per_period, replace=False))
                # print(problem[:,j, c])ws
                # print(y_reshaped[i,:,j,c])
                # x.append(problem[:,samples])
                # y.append(y_reshaped[i,:,samples])
                x = problem[samples].reshape(num_samples_in_total, 8, *x_reshaped.shape[-3:])
                y = y_problem[samples].reshape(num_samples_in_total, 2, *y_reshaped.shape[-3:])
                x = remove_borders_GP_predictions_lon_lat(x)
                assert x.shape[0] == y.shape[0]
                npaa_x.append(np.ascontiguousarray(x))
                npaa_y.append(np.ascontiguousarray(y))

    # In[9]:

    x = np.concatenate(x).reshape(-1, 8, num_samples_in_total, *x_reshaped.shape[-3:])

    # In[10]:

    y = np.concatenate(y).reshape(-1, 2, num_samples_in_total, *x_reshaped.shape[-3:])

    # In[12]:

    x.shape, y.shape


if __name__ == "__main__":
    main()
