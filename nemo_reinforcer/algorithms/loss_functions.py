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
        mult_prob_error = ((torch.exp(lp_error) * mask).sum() / mask.sum()).item()

        next_token_logits = next_token_logits[:, :-1]  # Remove last position's logits
        next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        next_tokens = data["input_ids"][:, 1:]  # Skip first token
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

        if mask.sum() > 0:
            actor_loss = masked_mean(torch.max(loss1, loss2), mask)
            loss = actor_loss + kl
        else:
            # disable this update since there are no valid tokens
            loss = loss1.view(-1)[0] * 0

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
            },
        )


class NLLLoss(LossFunction):
    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict,
        reduce_across_batch: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
        next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits

        # Gather the logprobs for the actual next tokens
        token_logprobs = logprobs.gather(
            dim=-1, index=next_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Only compute loss on generated tokens (not input tokens)
        # by applying the token_loss_mask (shifted by 1 since we're predicting next tokens)
        num_unmasked_tokens = torch.sum(mask, -1)
        num_unmasked_tokens[num_unmasked_tokens == 0] = 1

        if reduce_across_batch:
            num_unmasked_tokens = num_unmasked_tokens.sum().item()
            loss = (-torch.sum(token_logprobs * mask) / num_unmasked_tokens).item()
        else:
            loss = -torch.sum(token_logprobs * mask, dim=-1) / num_unmasked_tokens

        return loss, {
            "loss": loss,
            "num_unmasked_tokens": num_unmasked_tokens,
            "total_tokens": mask.numel(),
        }


class DPOLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    preference_loss_weight: float = 1.0
    sft_loss_weight: float = 0.0


class DPOLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    input_ids: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class DPOLossFn(LossFunction):
    def __init__(self, cfg: DPOLossConfig):
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.preference_loss_weight = cfg["preference_loss_weight"]
        self.sft_loss_weight = cfg["sft_loss_weight"]
        self.sft_loss = NLLLoss()

    def split_output_tensor(self, tensor: torch.Tensor):
        return tensor[::2], tensor[1::2]

    def preference_loss(
        self, next_token_logits: torch.Tensor, data: BatchedDataDict[DPOLossDataDict]
    ) -> torch.Tensor:
        ## TODO: make sure this token mask only includes the chosen / rejected responses
        ## and not prior assistant tokens
        ## TODO: there's some duplicate code here with the NLLLoss function. We should refactor
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
        next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits

        # Gather the logprobs for the actual next tokens
        token_logprobs = logprobs.gather(
            dim=-1, index=next_tokens.unsqueeze(-1)
        ).squeeze(-1)

        ref_logprobs = data["reference_policy_logprobs"][:, :-1]

        diff = (token_logprobs - ref_logprobs) * token_mask
        ## TODO: provide the option to average over tokens
        rewards = diff.sum(-1)

        ## TODO: make configurable. For now, do not average across sequences
        ## --> matches Aligner's default
        #if average_log_probs:
        #    # need to guard against divide by zero in case labels are all -100
        #    num_tokens_for_loss = token_mask.sum(-1)
        #    rewards = rewards / num_tokens_for_loss.clamp(min=1)

        ## ignore the batches whose sample_mask is 0
        #rewards = rewards[data["sample_mask"] == 1]

        #if len(rewards) == 0:
        #    return torch.tensor(0.0), torch.tensor(0.0)

        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected

        return -torch.nn.functional.logsigmoid(
            self.reference_policy_kl_penalty * rewards_delta
        ).mean(0), (rewards_chosen > rewards_rejected).float().mean(0)

    def __call__(
        self, next_token_logits: torch.Tensor, data: BatchedDataDict[DPOLossDataDict]
    ) -> Tuple[torch.Tensor, dict]:
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            sft_loss, _ = self.sft_loss(
                next_token_logits, data, reduce_across_batch=False
            )
            ## ignore the batches whose sample_mask is 0
            #sft_loss = sft_loss[data["sample_mask"] == 1]
            sft_loss_chosen, sft_loss_rejected = self.split_output_tensor(sft_loss)

            #if len(sft_loss_chosen) == 0:
            #    sft_loss_chosen = torch.tensor(0.0)

            ## average over the batch dimension
            sft_loss_chosen = sft_loss_chosen.mean(0)

        preference_loss, accuracy = self.preference_loss(next_token_logits, data)

        dpo_loss = (
            self.sft_loss_weight * sft_loss_chosen
            + self.preference_loss_weight * preference_loss
        )

        return dpo_loss, {
            "loss": dpo_loss.item(),
            "sft_loss": sft_loss_chosen.item(),
            "preference_loss": preference_loss.item(),
            "accuracy": accuracy.item(),
        }
