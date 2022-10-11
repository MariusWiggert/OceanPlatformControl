from typing import List, Dict
import gym
import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models import ModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.tf_utils import reduce_mean_ignore_inf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.tf.misc import normc_initializer
from torch import nn


class OceanDenseTorchModel(DQNTorchModel):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        common_units: List,
        common_activation: str,
        common_initializer_std: List,
        # dueling_units: List,
        # dueling_activation: str,
        # dueling_initializer_std: List,
        **kw
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        ##### Common Layers #####
        self.common_layers = []
        self.common_layers.append(tf.keras.layers.Input(shape=obs_space.shape, name="input_layer"))
        self.common_layers.append(tf.keras.layers.Flatten(name="flatten_layer")(self.common_layers[-1]))
        for i, units in enumerate(common_units):
            self.common_layers.append(
                tf.keras.layers.Dense(
                    units,
                    name=f"common_layer_{i}",
                    activation=get_activation_fn(common_activation[i]),
                    kernel_initializer=normc_initializer(common_initializer_std[i]),
                )(self.common_layers[-1])
            )
        self.base_model = tf.keras.Model(
            name='base_model',
            inputs=self.common_layers[0],
            outputs=self.common_layers[-1]
        )

        ##### Action Head #####
        # self.q_head_layers = []
        # self.q_head_layers.append(tf.keras.layers.Input(shape=(num_outputs,), name="input_layer"))
        # for i, units in enumerate(dueling_units + [action_space.n]):
        #     self.q_head_layers.append(
        #         tf.keras.layers.Dense(
        #             units,
        #             name=f"q_head_layer_{i}",
        #             activation=get_activation_fn(dueling_activation[i]),
        #             kernel_initializer=normc_initializer(dueling_initializer_std[i]),
        #         )(self.q_head_layers[-1])
        #     )
        # self.q_value_head = tf.keras.Model(
        #     name='q_head_model',
        #     inputs=self.q_head_layers[0],
        #     outputs=[
        #         self.q_head_layers[-1],
        #         tf.expand_dims(tf.ones_like(self.q_head_layers[-1]), -1),
        #         tf.expand_dims(tf.ones_like(self.q_head_layers[-1]), -1)
        #     ],
        # )

        ##### State Head #####
        # self.state_head_layers = []
        # self.state_head_layers.append(tf.keras.layers.Input(shape=(num_outputs,), name="input_layer"))
        # for i, units in enumerate(dueling_units + [1]):
        #     self.state_head_layers.append(
        #         tf.keras.layers.Dense(
        #             units,
        #             name=f"state_head_layer_{i}",
        #             activation=get_activation_fn(dueling_activation[i]),
        #             kernel_initializer=normc_initializer(dueling_initializer_std[i]),
        #         )(self.state_head_layers[-1])
        #     )
        # self.state_head_model = tf.keras.Model(
        #     name='state_head_model',
        #     inputs=self.state_head_layers[0],
        #     outputs=self.state_head_layers[-1]
        # )

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state