import numpy as np
import torch

from ocean_navigation_simulator.ocean_observer.models.CustomOceanCurrentsFromFiles import CustomOceanCurrentsFromFiles
from ocean_navigation_simulator.ocean_observer.models.OceanCurrentRunner import collate_fn, compute_burgers_loss

folder = "data_NN/data_validation_10_days_lstm_full/"

# %%
dataset_validation = CustomOceanCurrentsFromFiles([folder])
validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=32, collate_fn=collate_fn)
a = []
# Re = 507968
for i, (_, y) in enumerate(validation_loader):
    if not i % 100:
        print(f"index: {i, len(validation_loader)}")
    a.append(compute_burgers_loss(y))
    if i > 500:
        break

# %%
print(np.array(a).mean())

# %%
