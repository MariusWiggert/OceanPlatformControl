from typing import Tuple

import gym
import ray
import torch
from ray.rllib import Policy, SampleBatch
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
from ray.rllib.utils.torch_utils import FLOAT_MIN, huber_loss, l2_loss
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType
from torch import nn

from ocean_navigation_simulator.reinforcement_learning.OceanTorchModel import (
    OceanTorchModel,
)

"""
 This class modifies the rllib ApexDQN slightly s.t. that we can freely use custom models.
 The interaction of the policy with the model is overwritten.
 """


class CustomApexDQN(ApexDQN):
    def get_default_policy_class(self, config):
        if config["framework"] == "torch" and config.get("model").get("custom_model"):

            # Define custom model interaction for policy
            def custom_compute_q_values(
                policy: Policy,
                model: ModelV2,
                input_dict,
                state_batches=None,
                seq_lens=None,
                *args,
                **kwargs,
            ):
                (values, logits, logits), state = model(input_dict, state_batches or [], seq_lens)
                return values, logits, logits, state

            def custom_build_q_model_and_distribution(
                policy: Policy,
                obs_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                config: AlgorithmConfigDict,
            ) -> Tuple[ModelV2, TorchDistributionWrapper]:
                # Override model interaction in modeul 'dqn_torch_policy'
                import ray.rllib.algorithms.dqn.dqn_torch_policy as dqn_torch_policy

                dqn_torch_policy.compute_q_values = custom_compute_q_values

                model = OceanTorchModel(
                    obs_space,
                    action_space,
                    action_space.n,
                    config["model"],
                    "model",
                    **config["model"]["custom_model_config"],
                    num_atoms=config["num_atoms"],
                    v_min=config["v_min"],
                    v_max=config["v_max"],
                )
                policy.target_model = OceanTorchModel(
                    obs_space,
                    action_space,
                    action_space.n,
                    config["model"],
                    "model",
                    **config["model"]["custom_model_config"],
                    num_atoms=config["num_atoms"],
                    v_min=config["v_min"],
                    v_max=config["v_max"],
                )
                return model, get_torch_categorical_class_with_temperature(
                    config["categorical_distribution_temperature"]
                )

            # Build Policy from module 'dqn_torch_policy'
            import ray.rllib.algorithms.dqn.dqn_torch_policy as dqn_torch_policy
            from ray.rllib.algorithms.dqn.dqn_torch_policy import compute_q_values

            def custom_build_q_losses(policy: Policy, model, _, train_batch: SampleBatch) -> TensorType:
                """Constructs the loss for DQNTorchPolicy.

                    Args:
                        policy: The Policy to calculate the loss for.
                        model (ModelV2): The Model to calculate the loss for.
                        train_batch: The training data.

                    Returns:
                        TensorType: A single loss tensor.
                    """
                if not policy.config['cql']['use']:
                    return dqn_torch_policy.build_q_losses(policy, model, _, train_batch)
                else:
                    dqn_loss = dqn_torch_policy.build_q_losses(policy, model, _, train_batch)

                    # Batch x Actions
                    Q_a_s, _, _, _ = compute_q_values(
                        policy,
                        model,
                        {"obs": train_batch[SampleBatch.CUR_OBS]},
                        explore=False,
                        is_training=True,
                    )
                    cql_temp = policy.config["cql"]["temperature"]
                    cql_loss = cql_temp * torch.logsumexp(Q_a_s / cql_temp, dim=-1).mean() - Q_a_s.mean()

                    total_loss = dqn_loss + cql_loss

                    model.tower_stats["dqn_loss"] = model.tower_stats["q_loss"]
                    model.tower_stats["cql_loss"] = cql_loss
                    model.tower_stats["q_loss"] = total_loss

                    return total_loss

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
