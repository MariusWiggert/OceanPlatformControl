import tensorflow as tf
print("Tensorflow Devices:", tf.config.list_physical_devices('GPU'))


import torch
if torch.cuda.is_available():
    print('Torchh Device:', torch.cuda.get_device_name(0))
else:
    print("Torch cannot find GPU.")


from jax.lib import xla_bridge
print('Jax Backend:', xla_bridge.get_backend().platform)