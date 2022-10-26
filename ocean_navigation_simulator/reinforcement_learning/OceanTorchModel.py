import logging
from typing import Optional

import gym
import numpy as np
import torch
from ray.rllib.models.torch.misc import SlimConv2d, SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.typing import ModelConfigDict, TensorType
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

logger = logging.getLogger("ocean_torch_model")


class OceanCNN(nn.Module):
    """helper model to build convolution networks"""

    def __init__(self, name, input_size, channels, kernel, stride, padding):
        super().__init__()
        self.name = name

        layers = []

        for i in range(len(channels)):
            layers.append(
                SlimConv2d(
                    in_channels=input_size[2] if i == 0 else channels[i - 1],
                    out_channels=channels[i],
                    kernel=kernel[i],
                    stride=stride[i],
                    padding=padding[i],
                    # Defaulting these to nn.[..] will break soft torch import.
                    # initializer: Any = "default",
                    # activation_fn: Any = "default",
                    # bias_init: float = 0,
                )
            )

            layers.append(nn.Flatten())

        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)


class OceanDenseNet(nn.Module):
    """helper model to build dense networks"""

    def __init__(
        self,
        name,
        input_size,
        units,
        activation,
        initializer,
        input_activation=None,
        residual=False,
    ):
        super().__init__()
        self.name = name

        layers = [nn.Flatten()]

        if input_activation is not None:
            layers.append(get_activation_fn(input_activation, "torch")())

        for i in range(len(units)):
            if residual and i == len(units) - 1:
                _initializer = nn.init._no_grad_zero_
            elif initializer[i] == "xavier_uniform":
                _initializer = nn.init.xavier_uniform_
            elif isinstance(initializer[i], (int, float)):
                _initializer = normc_initializer(initializer[i])
            else:
                raise ValueError("Initalizer {initializer[i]} not valid.")

            layers.append(
                SlimFC(
                    in_size=int(np.product(input_size)) if i == 0 else units[i - 1],
                    out_size=units[i],
                    initializer=_initializer,
                    activation_fn=activation[i],
                )
            )

            if residual and i == len(units) - 1:
                linear = list(layers[-1].modules())[2]
                with torch.no_grad():
                    linear.bias.data[0] = torch.tensor([1])

        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)


class OceanTorchModel(TorchModelV2, nn.Module):
    """flexible Torch model used by RL. It is customizable with the configuration"""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        map: dict,
        meta: dict,
        joined: dict,
        dueling_heads: dict,
        **kwargs,
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        self.map_config = map
        self.meta_config = meta
        self.joined_config = joined
        self.dueling_heads_config = dueling_heads

        # Map Network
        self.map_in_shape = (
            obs_space[0].shape if isinstance(obs_space, gym.spaces.Tuple) else obs_space.shape
        )
        if self.map_config.get("channels"):
            self.map_model = OceanCNN(
                name="Map Preprocessing",
                input_size=self.map_in_shape,
                channels=self.map_config["channels"],
                kernel=self.map_config["kernel"],
                stride=self.map_config["stride"],
                padding=self.map_config["padding"],
            )
            self.map_out_shape = 1
        else:
            self.map_model = OceanDenseNet(
                name="Map Preprocessing",
                input_size=self.map_in_shape,
                units=self.map_config["units"],
                activation=self.map_config["activation"],
                initializer=self.map_config["initializer"],
            )
            self.map_out_shape = self.map_config["units"][-1]

        # Meta Network
        self.meta_in_shape = (1 if map["normalize"] else 0) + (
            obs_space[1].shape[0] if isinstance(obs_space, gym.spaces.Tuple) else 0
        )
        if self.meta_in_shape > 0 and (
            len(self.meta_config["units"]) > 0 or self.meta_config["input_activation"]
        ):
            self.meta_model = OceanDenseNet(
                name="Meta Preprocessing",
                input_size=self.meta_in_shape,
                units=self.meta_config["units"],
                activation=self.meta_config["activation"],
                initializer=self.meta_config["initializer"],
                input_activation=self.meta_config["input_activation"],
            )
            self.meta_out_shape = (
                self.meta_config["units"][-1]
                if not self.meta_config["input_activation"]
                else self.meta_in_shape
            )
        else:
            self.meta_out_shape = self.meta_in_shape

        # Joined Network
        self.joined_in_shape = self.map_out_shape + self.meta_out_shape
        if len(self.joined_config["units"]) > 0:
            self.joined_model = OceanDenseNet(
                name="Meta Preprocessing",
                input_size=self.joined_in_shape,
                units=self.joined_config["units"],
                activation=self.joined_config["activation"],
                initializer=self.joined_config["initializer"],
            )
            self.joined_out_shape = self.joined_config["units"][-1]
        else:
            self.joined_out_shape = self.joined_in_shape

        # Dueling Heads
        self.dueling_in_shape = self.joined_out_shape
        self.advantage_head = OceanDenseNet(
            name="Advantage Head",
            input_size=self.dueling_in_shape,
            units=self.dueling_heads_config["units"] + [action_space.n],
            activation=self.dueling_heads_config["activation"],
            initializer=self.dueling_heads_config["initializer"],
            residual=self.dueling_heads_config["residual"],
        )
        self.state_head = OceanDenseNet(
            name="State Head",
            input_size=self.dueling_in_shape,
            units=self.dueling_heads_config["units"] + [1],
            activation=self.dueling_heads_config["activation"],
            initializer=self.dueling_heads_config["initializer"],
        )

    def forward(self, map: TensorType, meta: TensorType = None):
        # Map Network
        if self.map_config["normalize"]:
            ttr_center = map[
                :,
                torch.div(map.shape[1] - 1, 2, rounding_mode="floor"),
                torch.div(map.shape[2] - 1, 2, rounding_mode="floor"),
                0,
            ]
            map = map - ttr_center[:, None, None, None]
        map_out = self.map_model(map)

        # Meta Network
        if self.meta_in_shape > 0:
            if self.map_config["normalize"] and meta:
                meta_in = torch.concat((meta, ttr_center[:, None]), 1)
            elif self.map_config["normalize"]:
                meta_in = ttr_center[:, None]
            else:
                meta_in = meta

            if len(self.meta_config["units"]) > 0 or self.meta_config["input_activation"]:
                meta_out = self.meta_model(meta_in)
            else:
                meta_out = meta_in

            joined_in = torch.concat((map_out, meta_out), 1)
        else:
            joined_in = map_out

        # Joined Network
        if len(self.joined_config["units"]) > 0 or self.joined_config["input_activation"]:
            joined_out = self.joined_model(joined_in)
        else:
            joined_out = joined_in

        # Calculate Dueling Heads
        advantage_out = self.advantage_head(joined_out)
        state_out = self.state_head(joined_out)

        # Reduce According to (9) in "Dueling Network Architectures for Deep Reinforcement Learning"
        advantages_mean = reduce_mean_ignore_inf(advantage_out, 1)
        values = state_out + advantage_out - advantages_mean[:, None]

        if torch.any(torch.isnan(values)):
            logger.error("Nan in torch model.")
            logger.error(f" Inputs: map:{map}, meta:{meta}")
            logger.error(
                f" Models: map_out:{map_out}, joined_out:{joined_out}, advantage_out:{advantage_out}, state_out:{state_out}"
            )
            logger.error(f" Output: advantages_mean:{advantages_mean}, values:{values}")

        return values

    def __call__(self, *args):
        with self.context():
            return self.forward(*args)


def reduce_mean_ignore_inf(x: TensorType, axis: Optional[int] = None) -> TensorType:
    """Same as torch.mean() but ignores -inf values.

    Args:
        x: The input tensor to reduce mean over.
        axis: The axis over which to reduce. None for all axes.

    Returns:
        The mean reduced inputs, ignoring inf values.
    """
    mask = torch.ne(x, float("-inf"))
    x_zeroed = torch.where(mask, x, torch.zeros_like(x))
    non_zero = torch.sum(mask.float(), axis)
    return torch.sum(x_zeroed, axis) / torch.max(non_zero, torch.ones_like(non_zero))
