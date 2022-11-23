import logging
from typing import Optional

import gym
import numpy as np
import torch
from ray.rllib.models.torch.misc import normc_initializer
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

    def __init__(
        self,
        name,
        input_shape,
        channels,
        kernels,
        strides,
        paddings,
        poolings,
        groups,
        dropouts,
    ):
        super().__init__()
        self.name = name

        layers = []
        for i in range(len(channels)):
            if paddings[i]:
                layers.append(nn.ZeroPad2d(paddings[i]))

            conv = nn.Conv2d(
                in_channels=input_shape[0] if i == 0 else channels[i - 1],
                out_channels=channels[i],
                kernel_size=kernels[i],
                stride=strides[i],
                groups=groups[i],
            )
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
            layers.append(conv)
            layers.append(nn.ReLU())
            if dropouts[i] > 0:
                layers.append(nn.Dropout2d(p=dropouts[i]))
            if poolings[i] is None:
                pass
            elif poolings[i][0] == "max":
                layers.append(torch.nn.MaxPool2d(kernel_size=poolings[i][1]))
            elif poolings[i][0] == "avg":
                layers.append(torch.nn.AvgPool2d(kernel_size=poolings[i][1]))
        layers.append(nn.Flatten())
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)


class OceanDenseNet(nn.Module):
    """helper model to build dense networks"""

    def __init__(
        self,
        name,
        input_shape,
        units,
        activations,
        initializers,
        dropouts,
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
                initializer = nn.init._no_grad_zero_
            elif initializers[i] == "xavier_uniform":
                initializer = nn.init.xavier_uniform_
            elif isinstance(initializers[i], (int, float)):
                initializer = normc_initializer(initializers[i])
            else:
                raise ValueError(f"Initializer {initializers[i]} not valid.")

            linear = nn.Linear(
                in_features=int(np.product(input_shape)) if i == 0 else units[i - 1],
                out_features=units[i],
                bias=True,
            )
            initializer(linear.weight)
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)

            activation_fn = get_activation_fn(activations[i], "torch")
            if activation_fn is not None:
                layers.append(activation_fn())

            if dropouts[i] > 0:
                layers.append(nn.Dropout(p=dropouts[i]))

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
        lstm: dict,
        dueling_heads: dict,

        num_atoms: int = 1,
        v_min: float = -10.0,
        v_max: float = 10.0,

        **kwargs,
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        self.map_config = map
        self.meta_config = meta
        self.joined_config = joined
        self.lstm_config = lstm
        self.dueling_heads_config = dueling_heads

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Map Network
        if obs_space.get("local_map", False):
            self.map_in_shape = obs_space.get("local_map").shape
            self.map_norm_shape = (
                self.map_in_shape[1] // self.map_config["groups"][0] if map["normalize"] else 0
            )

            if self.map_config.get("channels", False):
                self.map_model = OceanCNN(
                    name="Map Preprocessing CNN",
                    input_shape=self.map_in_shape,
                    channels=self.map_config["channels"],
                    kernels=self.map_config["kernels"],
                    strides=self.map_config["strides"],
                    paddings=self.map_config["paddings"],
                    poolings=self.map_config["poolings"],
                    groups=self.map_config["groups"],
                    dropouts=self.map_config.get(
                        "dropouts", [0] * len(self.map_config["channels"])
                    ),
                )
                self.map_out_shape = (
                    self.map_config["channels"][-1]
                    * int(np.prod(self.map_in_shape[1:]))
                    / self.map_config["poolings"][-1][1] ** 2
                )
            else:
                self.map_model = OceanDenseNet(
                    name="Map Preprocessing Dense",
                    input_shape=self.map_in_shape,
                    units=self.map_config["units"],
                    activations=self.map_config["activations"],
                    initializers=self.map_config["initializers"],
                    dropouts=self.map_config.get("dropouts", [0] * len(self.map_config["units"])),
                )
                if len(self.map_config["units"]) > 0:
                    self.map_out_shape = self.map_config["units"][-1]
                else:
                    self.map_out_shape = int(np.prod(self.map_in_shape))
        else:
            self.map_out_shape = 0
            self.map_norm_shape = 0

        # Meta Network
        self.meta_in_shape = self.map_norm_shape + (
            obs_space.get("meta").shape[0] if obs_space.get("meta", False) else 0
        )
        if self.meta_in_shape > 0 and (
            len(self.meta_config["units"]) > 0 or self.meta_config["input_activation"]
        ):
            self.meta_model = OceanDenseNet(
                name="Meta Preprocessing",
                input_shape=self.meta_in_shape,
                units=self.meta_config["units"],
                activations=self.meta_config["activations"],
                initializers=self.meta_config["initializers"],
                input_activation=self.meta_config["input_activation"],
                dropouts=self.meta_config.get("dropouts", [0] * len(self.meta_config["units"])),
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
                input_shape=self.joined_in_shape,
                units=self.joined_config["units"],
                activations=self.joined_config["activations"],
                initializers=self.joined_config["initializers"],
                dropouts=self.joined_config.get("dropouts", [0] * len(self.joined_config["units"])),
            )
            self.joined_out_shape = self.joined_config["units"][-1]
        else:
            self.joined_out_shape = self.joined_in_shape

        # LSTM

        # Dueling Heads
        self.dueling_in_shape = self.joined_out_shape
        self.advantage_out_shape = action_space.n * self.num_atoms
        self.advantage_head = OceanDenseNet(
            name="Advantage Head",
            input_shape=self.dueling_in_shape,
            units=self.dueling_heads_config["units"] + [self.advantage_out_shape],
            activations=self.dueling_heads_config["activations"],
            initializers=self.dueling_heads_config["initializers"],
            residual=self.dueling_heads_config["residual"],
            dropouts=self.dueling_heads_config.get(
                "dropouts", [0] * len(self.dueling_heads_config["units"] + [self.advantage_out_shape])
            ),
        )
        self.state_out_shape = self.num_atoms
        self.state_head = OceanDenseNet(
            name="State Head",
            input_shape=self.dueling_in_shape,
            units=self.dueling_heads_config["units"] + [self.state_out_shape],
            activations=self.dueling_heads_config["activations"],
            initializers=self.dueling_heads_config["initializers"],
            dropouts=self.dueling_heads_config.get(
                "dropouts", [0] * len(self.dueling_heads_config["units"] + [self.state_out_shape])
            ),
        )

    def forward(self, local_map: TensorType = None, meta: TensorType = None):
        # Map Network
        if self.map_config["normalize"]:
            # batch, time*var, x, y
            vars = torch.div(
                local_map.shape[1], self.map_config["groups"][0], rounding_mode="floor"
            )
            ttr_center = local_map[
                :,
                :vars:,
                torch.div(local_map.shape[2] - 1, 2, rounding_mode="floor"),
                torch.div(local_map.shape[3] - 1, 2, rounding_mode="floor"),
            ]
            local_map[:, :vars:, :, :] = local_map[:, :vars:, :, :] - ttr_center[:, :, None, None]

        # Meta Network
        if self.meta_in_shape > 0:
            if self.map_config["normalize"] and meta is not None:
                meta_in = torch.concat((meta, ttr_center), 1)
            elif self.map_config["normalize"]:
                meta_in = ttr_center
            else:
                meta_in = meta

            if len(self.meta_config["units"]) > 0 or self.meta_config["input_activation"]:
                meta_out = self.meta_model(meta_in)
            else:
                meta_out = meta_in

        if local_map is not None:
            map_out = self.map_model(local_map)

            if self.meta_in_shape > 0:
                joined_in = torch.concat((map_out, meta_out), 1)
            else:
                joined_in = map_out

        else:
            joined_in = meta_out

        # Joined Network
        if len(self.joined_config["units"]) > 0 or self.joined_config["input_activation"]:
            joined_out = self.joined_model(joined_in)
        else:
            joined_out = joined_in

        # LSTM
        lstm_out = joined_out

        # Calculate Dueling Heads
        action_scores = self.advantage_head(lstm_out)
        state_scores = self.state_head(lstm_out)

        if self.num_atoms > 1:
            # Distributional Q-learning uses a discrete support z
            # to represent the action value distribution
            z = torch.arange(0.0, self.num_atoms, dtype=torch.float32).to(
                action_scores.device
            )
            z = self.v_min + z * (self.v_max - self.v_min) / float(self.num_atoms - 1)

            support_logits_per_action = torch.reshape(
                action_scores, shape=(-1, self.action_space.n, self.num_atoms)
            )
            support_prob_per_action = nn.functional.softmax(
                support_logits_per_action, dim=-1
            )
            action_scores = torch.sum(z * support_prob_per_action, dim=-1)
            return action_scores, z, support_logits_per_action
        else:
            # Reduce According to (9) in "Dueling Network Architectures for Deep Reinforcement Learning"
            advantages_mean = reduce_mean_ignore_inf(action_scores, 1)
            values = state_scores + action_scores - advantages_mean[:, None]
            return values

    def __call__(self, *args, **kwargs):
        with self.context():
            return self.forward(*args, **kwargs)


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
