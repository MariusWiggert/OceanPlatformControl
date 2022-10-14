from typing import List, Dict, Any
import gym
import numpy as np
import tensorflow as tf
import torch
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import reduce_mean_ignore_inf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.misc import normc_initializer
from torch import nn


# Documentation:
#   https://docs.ray.io/en/latest/rllib/package_ref/models.html
#   https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
#   https://docs.ray.io/en/latest/rllib/rllib-models.html#custom-models-implementing-your-own-forward-logic
# Usage:
#   https://github.com/ray-project/ray/blob/releases/1.13.0/rllib/agents/dqn/distributional_q_tf_model.py
#   https://github.com/ray-project/ray/blob/releases/1.13.0/rllib/models/tf/tf_modelv2.py
#   https://github.com/ray-project/ray/blob/releases/1.13.0/rllib/models/modelv2.py
#   https://github.com/ray-project/ray/blob/releases/1.13.0/rllib/agents/dqn/dqn_tf_policy.py

class OceanDenseNet(nn.Module):
    def __init__(self, input_size, units, activation, initializer_std):
        # print('-- OceanDenseNet.__init__ --')
        # print('input_size', input_size, 'units', units, 'activation', activation, 'initializer_std', initializer_std)
        # print('')
        # print('')

        super().__init__()

        layers = [nn.Flatten()]

        for i in range(len(units)):
            layers.append(
                SlimFC(
                    in_size=int(np.product(input_size)) if i==0 else units[i-1],
                    out_size=units[i],
                    initializer=normc_initializer(initializer_std[i]) if initializer_std[i] else None,
                    activation_fn=activation[i],
                )
            )

        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)

class OceanTorchModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,

        map: dict,
        dueling_heads: dict,

        **kwargs
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # print(' -- OceanTorchModel.__init__ --')
        # print('obs_space', obs_space, 'action_space', action_space, 'num_outputs', num_outputs, 'model_config', model_config, 'name', name)
        # print('')
        # print('map', map, 'dueling_heads', dueling_heads)
        # print('')
        # print(kwargs)
        # print('')
        # print('')


        # Map Network
        self.map_model = OceanDenseNet(
            input_size=obs_space[0].shape,
            units=map['units'],
            activation=map['activation'],
            initializer_std=map['initializer_std'],
        )

        # Dueling Heads
        joined_shape = map['units'][-1] + 1 + obs_space[1].shape[0]
        self.advantage_head = OceanDenseNet(
            input_size=joined_shape,
            units=dueling_heads['units'] + [action_space.n],
            activation=dueling_heads['activation'],
            initializer_std=dueling_heads['initializer_std'],
        )
        self.state_head = OceanDenseNet(
            input_size=joined_shape,
            units=dueling_heads['units'] + [1],
            activation=dueling_heads['activation'],
            initializer_std=dueling_heads['initializer_std'],
        )

    def forward(self, map: TensorType, meta: TensorType):
        # print(' -- OceanTorchModel.forward --')
        # print('map.shape', map.shape, 'meta.shape', meta.shape)
        # print('')

        ttr_center = map[:, (map.shape[1]-1) // 2, (map.shape[2]-1) // 2, 0]
        normalized_map = map - ttr_center[:, None, None, None]

        # print(normalized_map)

        map_out = self.map_model(normalized_map)

        # Join: Map + Normalization + Meta
        # print(map_out.shape, ttr_center.shape, meta.shape)

        joined = torch.concat((map_out, ttr_center[:, None], meta), 1)

        # Calculate Dueling Heads
        advantage_out = self.advantage_head(joined)
        state_out = self.state_head(joined)


        # Reduce According to (9) in "Dueling Network Architectures for Deep Reinforcement Learning"
        advantages_mean = reduce_mean_ignore_inf(advantage_out, 1)
        # print(advantage_out.shape, state_out.shape, advantages_mean.shape)
        values = state_out + advantage_out - advantages_mean[:, None]

        # print(values)

        return values

    def __call__(self, **kwargs):
        with self.context():
            return self.forward(**kwargs)