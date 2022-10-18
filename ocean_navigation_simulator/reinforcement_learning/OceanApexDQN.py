from typing import Tuple

import gym
import ray
import torch
from ray.rllib import Policy
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper,
    get_torch_categorical_class_with_temperature,
)
from ray.rllib.policy import build_policy_class
from ray.rllib.policy.torch_mixins import (
    LearningRateSchedule,
    TargetNetworkMixin,
)
from ray.rllib.utils.typing import AlgorithmConfigDict
from ray.tune.registry import RLLIB_MODEL, _global_registry


class OceanApexDQN(ApexDQN):
    """
    This class modifies the rllib ApexDQN slightly s.t. that we can freely use custom models.
    The interaction of the policy with the model is overwritten.
    """

    def get_default_policy_class(self, config):
        if config["framework"] == "torch" and config.get("model").get("custom_model"):
            # Define custom model interaction for policy
            def custom_compute_q_values(policy: Policy, model: ModelV2, input_dict, **kwargs):
                if isinstance(input_dict["obs"], tuple):
                    values = model(*input_dict["obs"])
                else:
                    values = model(input_dict["obs"])
                logits = torch.unsqueeze(torch.ones_like(values), -1)

                return values, logits, logits, []

            def custom_build_q_model_and_distribution(
                policy: Policy,
                obs_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                config: AlgorithmConfigDict,
            ) -> Tuple[ModelV2, TorchDistributionWrapper]:
                # Override model interaction in modeul 'dqn_torch_policy'
                import ray.rllib.algorithms.dqn.dqn_torch_policy as dqn_torch_policy

                dqn_torch_policy.compute_q_values = custom_compute_q_values

                model_cls = _global_registry.get(RLLIB_MODEL, config["model"]["custom_model"])
                model = model_cls(
                    obs_space,
                    action_space,
                    action_space.n,
                    config["model"],
                    "model",
                    **config["model"]["custom_model_config"]
                )
                policy.target_model = model_cls(
                    obs_space,
                    action_space,
                    action_space.n,
                    config["model"],
                    "model",
                    **config["model"]["custom_model_config"]
                )
                return model, get_torch_categorical_class_with_temperature(
                    config["categorical_distribution_temperature"]
                )

            # Build Policy from module 'dqn_torch_policy'
            import ray.rllib.algorithms.dqn.dqn_torch_policy as dqn_torch_policy

            return build_policy_class(
                name="CustomDQNTorchPolicy",
                framework="torch",
                loss_fn=dqn_torch_policy.build_q_losses,
                get_default_config=lambda: ray.rllib.algorithms.dqn.dqn.DEFAULT_CONFIG,
                make_model_and_action_dist=custom_build_q_model_and_distribution,
                action_distribution_fn=dqn_torch_policy.get_distribution_inputs_and_class,
                stats_fn=dqn_torch_policy.build_q_stats,
                postprocess_fn=dqn_torch_policy.postprocess_nstep_and_prio,
                optimizer_fn=dqn_torch_policy.adam_optimizer,
                extra_grad_process_fn=dqn_torch_policy.grad_process_and_td_error_fn,
                extra_learn_fetches_fn=dqn_torch_policy.concat_multi_gpu_td_errors,
                extra_action_out_fn=dqn_torch_policy.extra_action_out_fn,
                before_init=dqn_torch_policy.setup_early_mixins,
                before_loss_init=dqn_torch_policy.before_loss_init,
                mixins=[
                    TargetNetworkMixin,
                    dqn_torch_policy.ComputeTDErrorMixin,
                    LearningRateSchedule,
                ],
            )
        else:
            return ApexDQN.get_default_policy_class(self, config)
