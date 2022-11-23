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
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType

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

            def custom_compute_q_values(policy: Policy, model: ModelV2, input_dict, **kwargs):


                if config["num_atoms"] > 1:
                    z, support_logits_per_action = model(**input_dict["obs"])

                    support_logits_per_action_mean = torch.mean(
                        support_logits_per_action, dim=1
                    )
                    support_logits_per_action_centered = (
                        support_logits_per_action
                        - torch.unsqueeze(support_logits_per_action_mean, dim=1)
                    )
                    support_logits_per_action = (
                        torch.unsqueeze(state_score, dim=1) + support_logits_per_action_centered
                    )
                    support_prob_per_action = nn.functional.softmax(
                        support_logits_per_action, dim=-1
                    )
                    value = torch.sum(z * support_prob_per_action, dim=-1)

                    return value, support_logits_per_action, support_prob_per_action, []
                else:
                    values = model(**input_dict["obs"])
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

            from ray.rllib.algorithms.dqn.dqn_torch_policy import (
                QLoss,
                compute_q_values,
                FLOAT_MIN,
                huber_loss,
                l2_loss,
                PRIO_WEIGHTS,
                F,
            )

            def custom_build_q_losses(
                policy: Policy, model, _, train_batch: SampleBatch
            ) -> TensorType:
                """Constructs the loss for DQNTorchPolicy.

                Args:
                    policy: The Policy to calculate the loss for.
                    model (ModelV2): The Model to calculate the loss for.
                    train_batch: The training data.

                Returns:
                    TensorType: A single loss tensor.
                """

                config = policy.config
                # Q-network evaluation.
                q_t, q_logits_t, q_probs_t, _ = compute_q_values(
                    policy,
                    model,
                    {"obs": train_batch[SampleBatch.CUR_OBS]},
                    explore=False,
                    is_training=True,
                )

                # Target Q-network evaluation.
                q_tp1, q_logits_tp1, q_probs_tp1, _ = compute_q_values(
                    policy,
                    policy.target_models[model],
                    {"obs": train_batch[SampleBatch.NEXT_OBS]},
                    explore=False,
                    is_training=True,
                )

                # Q scores for actions which we know were selected in the given state.
                one_hot_selection = F.one_hot(
                    train_batch[SampleBatch.ACTIONS].long(), policy.action_space.n
                )
                q_t_selected = torch.sum(
                    torch.where(q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=q_t.device))
                    * one_hot_selection,
                    1,
                )
                q_logits_t_selected = torch.sum(
                    q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1
                )

                # compute estimate of best possible value starting from state at t + 1
                if config["double_q"]:
                    (
                        q_tp1_using_online_net,
                        q_logits_tp1_using_online_net,
                        q_dist_tp1_using_online_net,
                        _,
                    ) = compute_q_values(
                        policy,
                        model,
                        {"obs": train_batch[SampleBatch.NEXT_OBS]},
                        explore=False,
                        is_training=True,
                    )
                    q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
                    q_tp1_best_one_hot_selection = F.one_hot(
                        q_tp1_best_using_online_net, policy.action_space.n
                    )
                    q_tp1_best = torch.sum(
                        torch.where(
                            q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
                        )
                        * q_tp1_best_one_hot_selection,
                        1,
                    )
                    q_probs_tp1_best = torch.sum(
                        q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
                    )
                else:
                    q_tp1_best_one_hot_selection = F.one_hot(
                        torch.argmax(q_tp1, 1), policy.action_space.n
                    )
                    q_tp1_best = torch.sum(
                        torch.where(
                            q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
                        )
                        * q_tp1_best_one_hot_selection,
                        1,
                    )
                    q_probs_tp1_best = torch.sum(
                        q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
                    )

                loss_fn = huber_loss if policy.config["td_error_loss_fn"] == "huber" else l2_loss

                q_loss = QLoss(
                    q_t_selected,
                    q_logits_t_selected,
                    q_tp1_best,
                    q_probs_tp1_best,
                    train_batch[PRIO_WEIGHTS],
                    train_batch[SampleBatch.REWARDS],
                    train_batch[SampleBatch.DONES].float(),
                    config["gamma"],
                    config["n_step"],
                    config["num_atoms"],
                    config["v_min"],
                    config["v_max"],
                    loss_fn,
                )

                # Store values for stats function in model (tower), such that for
                # multi-GPU, we do not override them during the parallel loss phase.
                model.tower_stats["td_error"] = q_loss.td_error
                # TD-error tensor in final stats
                # will be concatenated and retrieved for each individual batch item.
                model.tower_stats["q_loss"] = q_loss

                return q_loss.loss

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
