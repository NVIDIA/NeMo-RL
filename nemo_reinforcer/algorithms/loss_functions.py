# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Tuple, TypedDict

import torch

from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.algorithms.utils import (
    calculate_kl_penalty_joschu2020,
    masked_mean,
)

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.models.dtensor.parallelize import (
    get_logprobs_from_vocab_parallel_logits,
)


class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    ratio_eps_min: float
    ratio_eps_max: float


class ClippedPGLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    input_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    __extra__: Any


class ClippedPGLossFn(LossFunction):
    """Generalized Clipped Policy Gradient loss function w/ KL regularization.

    This implements:

    - PPO (Clipped) - https://arxiv.org/abs/1707.06347
    - GRPO - https://arxiv.org/abs/2402.03300
    - REINFORCE/RLOO (set disable_ppo_ratio = True and ignores ratio_eps) - https://arxiv.org/abs/2402.14740

    Formula:
    L(θ) = E_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
    - A_t is the advantage estimate
    - ε is the clip parameter (ratio_eps)
        - As proposed in the DAPO paper (https://arxiv.org/pdf/2503.14476),
          we allow setting a distinct minimum and maximum value for the clip parameter (set to the same value for PPO/GRPO/etc.)
            - ratio_eps_min: minimum value for the clip parameter
            - ratio_eps_max: maximum value for the clip parameter
    - β is the KL penalty coefficient (reference_policy_kl_penalty)
    - KL(π_θ || π_ref) is the KL divergence between the current policy and reference policy (Schulman Approx.)

    For REINFORCE/RLOO (when disable_ppo_ratio=True), the formula simplifies to:
    L(θ) = E_t [ π_θ(a_t|s_t) * A_t ] - β * KL(π_θ || π_ref)

    Due to potential numerical instability, we cast the logits to float32 before computing the loss.
    """

    def __init__(self, cfg: ClippedPGLossConfig):
        self.ratio_eps_min = cfg["ratio_eps_min"]
        self.ratio_eps_max = cfg["ratio_eps_max"]
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict[ClippedPGLossDataDict],
    ) -> Tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss function."""
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        advantages = data["advantages"][:, 1:]
        prev_logprobs = data["prev_logprobs"][:, 1:]
        generation_logprobs = data["generation_logprobs"][:, 1:]
        reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]

        mask = token_mask * sample_mask.unsqueeze(-1)

        lp_error = torch.abs(generation_logprobs - prev_logprobs)  # noqa: F841  (precommit ignore for now)
        mult_prob_error = masked_mean(torch.exp(lp_error), mask).item()

        next_token_logits = next_token_logits.to(torch.float32)

        if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            curr_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"]
            )
        else:
            next_token_logits = next_token_logits[
                :, :-1
            ]  # Remove last position's logits
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
            curr_logprobs = next_token_logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        # Calculate KL regularization.
        if self.reference_policy_kl_penalty != 0:
            kl = self.reference_policy_kl_penalty * calculate_kl_penalty_joschu2020(
                logprobs_policy=curr_logprobs,
                logprobs_reference=reference_policy_logprobs,
            )
            kl = masked_mean(kl, mask)
        else:
            kl = 0

        # Calculate clipped loss function if ppo ratio is enabled.
        if not self.disable_ppo_ratio:
            ratios = (curr_logprobs - prev_logprobs).exp()
            ratios_clamped = ratios.clamp(
                1.0 - self.ratio_eps_min, 1.0 + self.ratio_eps_max
            )
        else:
            ratios = curr_logprobs
            ratios_clamped = curr_logprobs

        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped

        actor_loss = masked_mean(torch.max(loss1, loss2), mask)
        loss = actor_loss + kl
        with torch.no_grad():
            probs_ratio = masked_mean(ratios.detach(), mask).item()
            probs_ratio_clamped = masked_mean(ratios_clamped.detach(), mask).item()

        return (
            loss,
            {
                "loss": loss.item(),
                "probs_ratio": probs_ratio,
                "probs_ratio_clamped": probs_ratio_clamped,
                "kl_penalty": kl.item() / self.reference_policy_kl_penalty if kl else 0,
                "token_mult_prob_error": mult_prob_error,
                "num_valid_samples": sample_mask.sum().item(),
            },
        )


class NLLLoss(LossFunction):
    """Negative Log Likelihood Loss function."""

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict,
        dpo_loss: bool = False,
        dpo_average_log_probs: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        next_token_logits = next_token_logits.to(torch.float32)

        # Gather the logprobs for the actual next tokens
        if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"]
            )
        else:
            next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        if dpo_loss:
            ## shape: [batch_size]
            num_unmasked_tokens = torch.sum(mask, -1)
            ## multiply by sample_mask to zero out invalid samples
            loss = -torch.sum(token_logprobs * mask, dim=-1)
            if dpo_average_log_probs:
                loss = loss / num_unmasked_tokens.clamp(min=1)
        else:
            ## single scalar loss
            # Only compute loss on generated tokens (not input tokens)
            # by applying the token_loss_mask
            num_unmasked_tokens = torch.sum(mask)
            if num_unmasked_tokens == 0:
                # prevent division by zero
                num_unmasked_tokens = torch.tensor(1)
            loss = -torch.sum(token_logprobs * mask) / num_unmasked_tokens
            num_unmasked_tokens = num_unmasked_tokens.item()

        return loss, {
            "loss": loss.item() if loss.ndim == 0 else loss,
            "num_unmasked_tokens": num_unmasked_tokens,
            "total_tokens": mask.numel(),
            "num_valid_samples": sample_mask.sum().item(),
        }


class DPOLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    preference_loss_weight: float = 1.0
    sft_loss_weight: float = 0.0
    preference_average_log_probs: bool = False
    sft_average_log_probs: bool = False


class DPOLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    input_ids: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class DPOLossFn(LossFunction):
    """Direct Preference Optimization (DPO) loss function.

    This loss function implements the DPO algorithm as described in:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (https://arxiv.org/abs/2305.18290)

    The loss combines two main components:
    1. Preference Loss: Optimizes the model to prefer chosen responses over rejected ones
    2. SFT Loss (optional): Auxiliary supervised fine-tuning loss on chosen responses

    The total loss is computed as:
    L(θ) = w_p * L_pref(θ) + w_s * L_sft(θ)

    where:
    - w_p is the preference_loss_weight
    - w_s is the sft_loss_weight
    - L_pref(θ) is the preference loss term
    - L_sft(θ) is the supervised fine-tuning loss term

    The preference loss term is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is the reference_policy_kl_penalty
    - r_chosen and r_rejected are the rewards for chosen and rejected responses
    - The rewards are computed as the sum of log probability differences between
      the current policy and reference policy

    If preference_average_log_probs is True, the rewards are averaged over tokens:
    r = (1/n) * Σ_t (log π_θ(a_t|s_t) - log π_ref(a_t|s_t))

    Otherwise, the rewards are summed over tokens.

    The SFT loss term is a standard negative log likelihood loss on the chosen responses.
    If sft_average_log_probs is True, the loss is averaged over tokens.

    Args:
        cfg (DPOLossConfig): Configuration dictionary containing:
            - reference_policy_kl_penalty (float): Strength of the KL penalty term (β)
            - preference_loss_weight (float): Weight for the preference loss term (w_p)
            - sft_loss_weight (float): Weight for the SFT loss term (w_s)
            - preference_average_log_probs (bool): Whether to average log probs across tokens in preference loss
            - sft_average_log_probs (bool): Whether to average log probs across tokens in SFT loss

    Returns:
        Tuple[torch.Tensor, dict]: A tuple containing:
            - The total loss value
            - A dictionary with metrics including:
                - loss: Total loss value
                - sft_loss: SFT loss component
                - preference_loss: Preference loss component
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    def __init__(self, cfg: DPOLossConfig):
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.preference_loss_weight = cfg["preference_loss_weight"]
        self.sft_loss_weight = cfg["sft_loss_weight"]
        self.preference_average_log_probs = cfg["preference_average_log_probs"]
        self.sft_average_log_probs = cfg["sft_average_log_probs"]
        self.sft_loss = NLLLoss()

    def split_output_tensor(self, tensor: torch.Tensor):
        return tensor[::2], tensor[1::2]

    def preference_loss(
        self, next_token_logits: torch.Tensor, data: BatchedDataDict[DPOLossDataDict]
    ) -> torch.Tensor:
        ## TODO(@ashors): there's some duplicate code here with the NLLLoss function. We should refactor
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]

        next_token_logits = next_token_logits.to(torch.float32)
        if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"]
            )
        else:
            next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        ref_logprobs = data["reference_policy_logprobs"][:, :-1]

        diff = (token_logprobs - ref_logprobs) * token_mask

        rewards = diff.sum(-1)
        if self.preference_average_log_probs:
            rewards = rewards / token_mask.sum(-1).clamp(min=1)

        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(
                self.reference_policy_kl_penalty * rewards_delta
            )
            * sample_mask[::2]
        )  ## zero out invalid samples

        return (
            masked_mean(per_sample_loss, sample_mask[::2]),
            (rewards_chosen > rewards_rejected).float().mean(0),
            masked_mean(rewards_chosen, sample_mask[::2]),
            masked_mean(rewards_rejected, sample_mask[1::2]),
        )

    def __call__(
        self, next_token_logits: torch.Tensor, data: BatchedDataDict[DPOLossDataDict]
    ) -> Tuple[torch.Tensor, dict]:
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            sft_loss, _ = self.sft_loss(
                next_token_logits,
                data,
                dpo_loss=True,
                dpo_average_log_probs=self.sft_average_log_probs,
            )
            sft_loss_chosen, sft_loss_rejected = self.split_output_tensor(sft_loss)
            sft_loss_chosen = masked_mean(sft_loss_chosen, data["sample_mask"][::2])

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self.preference_loss(next_token_logits, data)

        dpo_loss = (
            self.sft_loss_weight * sft_loss_chosen
            + self.preference_loss_weight * preference_loss
        )

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = data["sample_mask"].sum() / 2

        return dpo_loss, {
            "loss": dpo_loss.item(),
            "sft_loss": sft_loss_chosen.item(),
            "preference_loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }
