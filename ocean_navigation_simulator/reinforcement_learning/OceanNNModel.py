import gym
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict

tf1, tf, tfv = try_import_tf()


# Documentation: https://docs.ray.io/en/latest/rllib/package_ref/models.html
# Example: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/parametric_actions_model.py

class OceanNNModel(DistributionalQTFModel):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        q_hiddens=(256,),
        add_layer_norm: bool = False,
        use_custom: bool = False,
        **kw
    ):
        super(OceanNNModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw
        )
        # self.layers = []
        # self.layers.append(tf.keras.layers.Input(shape=obs_space.shape, name="observations"))
        #
        # print('num_outputs:', num_outputs)
        # print('obs_space:', obs_space)
        #
        # for index, units in enumerate(model_config['custom_model_config']['hidden_units']):
        #     self.layers.append(tf.keras.layers.Dense(
        #         units,
        #         activation=tf.nn.relu,
        #         kernel_initializer=normc_initializer(1.0),
        #         name=f"fc_layer_{index}",
        #     )(self.layers[index].flatten()))
        #
        # layer_out = tf.keras.layers.Dense(
        #     num_outputs,
        #     name="layer_out",
        #     activation=None,
        #     kernel_initializer=normc_initializer(0.01),
        # )(self.layers[-1])
        #
        # self.base_model = tf.keras.Model(self.layers[0], layer_out, name="OceanNNModel")
        layers = []
        layers.append(tf.keras.layers.Input(shape=obs_space.shape, name="input_layer"))
        layers.append(tf.keras.layers.Flatten(name="flatten_layer")(layers[-1]))

        for index, units in enumerate(q_hiddens):
            layers.append(
                tf.keras.layers.Dense(
                    units,
                    name=f"dense_layer_{index}",
                    activation=tf.nn.relu,
                    kernel_initializer=normc_initializer(1.0),
                )(layers[-1])
            )
        layers.append(
            tf.keras.layers.Dense(
                num_outputs,
                name="output_layer",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0),
            )(layers[-1])
        )
        self.base_model = tf.keras.Model(layers[0], layers[-1])

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

    def metrics(self):
        return {"foo": tf.constant(42.0)}