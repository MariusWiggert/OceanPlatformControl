from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class OceanNNModel(DistributionalQModel):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        pass

    def forward(self, input_dict, state, seq_lens):
        pass

    def value_function(self):
        pass