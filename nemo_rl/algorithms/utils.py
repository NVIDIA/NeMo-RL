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
import os
import random
import warnings
from functools import wraps
from typing import Callable, Optional, Tuple, TypedDict

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.data import hf_datasets
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.dtensor.parallelize import (
    get_logprobs_from_vocab_parallel_logits,
)
from nemo_rl.models.policy import PolicyConfig, TokenizerConfig
from nemo_rl.models.policy.hf_policy import HfPolicy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer


def calculate_kl_penalty_joschu2020(
    logprobs_policy: torch.Tensor, logprobs_reference: torch.Tensor
) -> torch.Tensor:
    """Calculates a per-token estimate of the KL Divergence between two log_probs.

    From Schulman 2020, always positive.

    logprobs_policy:    torch.Tensor (b, s)
    logprobs_reference: torch.Tensor (b, s)
    """
    r = logprobs_reference - logprobs_policy
    return torch.exp(r) - r - 1


def calculate_baseline_and_std_per_prompt(
    prompts: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    leave_one_out_baseline: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to compute a baseline for each (prompt, response) pair in the batch.

    The same baseline is calculated for each prompt. Samples set to 0 in 'valid_mask'
    are not included in the baseline calculation.

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    rewards:    tensor (b,)       Float-valued rewards. May be on any device
    valid_mask: tensor (b,)       Vector of 0/1, where 0 is to ignore and 1 is to keep
    leave_one_out_baseline: bool  Compute an unbiased baseline by leaving out the sample that
                                  the baseline is for (from RLOO https://arxiv.org/abs/2402.14740)

    Returns:
    tensor (b,), tensor (b,) of baselines and std on the same device as 'rewards'
    """
    unique_prompts = torch.unique(prompts, dim=0)

    baseline = torch.zeros_like(rewards)
    sq_baseline = torch.zeros_like(rewards)
    device_ordinal = rewards.get_device()
    if device_ordinal == -1:
        reward_device = torch.device("cpu")
    else:
        reward_device = torch.device(reward_device)

    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[
            is_matching_prompt
        ]

        if leave_one_out_baseline:
            baseline_mask_matrix = (1 - torch.eye(len(prompt_idx))).to(reward_device)
        else:
            baseline_mask_matrix = torch.ones((len(prompt_idx), len(prompt_idx))).to(
                reward_device
            )

        if valid_mask[prompt_idx].sum() <= 1:
            # Ignore sample: there are no valid responses, so set baseline equal to reward
            # to ignore it in the loss computation
            baseline[prompt_idx] = rewards[prompt_idx]
        else:
            num_valid = valid_mask[prompt_idx].float().sum() - int(
                leave_one_out_baseline
            )
            prompt_baseline = (
                torch.matmul(
                    baseline_mask_matrix, rewards[prompt_idx] * valid_mask[prompt_idx]
                )
                / num_valid
            )
            prompt_baseline_square = (
                torch.matmul(
                    baseline_mask_matrix,
                    (rewards[prompt_idx] ** 2) * valid_mask[prompt_idx],
                )
                / num_valid
            )

            baseline[prompt_idx] = prompt_baseline
            sq_baseline[prompt_idx] = prompt_baseline_square

    std = (sq_baseline - baseline.square()).sqrt().nan_to_num(0)
    return baseline, std


def surpress_user_warnings(f):  # type: ignore
    @wraps(f)
    def wrapper(*args, **kwargs):  # type: ignore
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            output = f(*args, **kwargs)
        return output

    return wrapper


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    global_normalization_factor: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Computes the mean of a microbatch, using a global statistic as the normalization factor."""
    normalization_factor = (
        torch.sum(mask, dim=dim)
        if global_normalization_factor is None
        else global_normalization_factor
    )
    return torch.sum(values * mask, dim=dim) / (normalization_factor + 1e-8)


def get_logprobs(
    data: BatchedDataDict, next_token_logits: torch.Tensor
) -> torch.Tensor:
    """Get the log probabilities for the (potentially vocab parallel) actual next tokens.

    This function handles gathering log probabilities for both FSDP1 and dtensor.

    Args:
        data (BatchedDataDict): Dictionary containing input_ids tensor with token indices
        next_token_logits (torch.Tensor): Tensor of logits for next token predictions

    Returns:
        torch.Tensor: Log probabilities of the actual next tokens that occurred in the sequence
    """
    if isinstance(next_token_logits, torch.distributed.tensor.DTensor):
        token_logprobs = get_logprobs_from_vocab_parallel_logits(
            next_token_logits, data["input_ids"]
        )
    else:
        next_token_logits_wo_last = next_token_logits[
            :, :-1
        ]  # Remove last position's logits
        next_token_logprobs = torch.nn.functional.log_softmax(
            next_token_logits_wo_last, dim=-1
        )
        next_tokens = data.get("input_ids")[:, 1:].cuda()  # Skip first token
        token_logprobs = next_token_logprobs.gather(
            dim=-1, index=next_tokens.unsqueeze(-1)
        ).squeeze(-1)

    return token_logprobs


def set_seed(seed: int) -> None:
    """Sets the seed for python, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(tokenizer_config: TokenizerConfig) -> PreTrainedTokenizerBase:
    """Get the tokenizer and set pad token to eos token if it is not already set.

    This function initializes a tokenizer from the Hugging Face transformers library
    and configures it with appropriate chat templates and padding tokens.

    Args:
        tokenizer_config: A dictionary containing tokenizer configuration.
            Required keys:
                - name: The name or path of the pretrained tokenizer
            Optional keys:
                - chat_template: The chat template to use. Can be:
                    - None: Uses a passthrough template that just returns message content
                    - "default": Uses the tokenizer's default template
                    - A custom jinja2 template string
                    If not specified, the tokenizer's default template will be used.

    Returns:
        PreTrainedTokenizerBase: The configured tokenizer instance

    Examples:
        ```{doctest}
        >>> from transformers import AutoTokenizer
        >>> from nemo_rl.algorithms.utils import get_tokenizer
        >>> # not specifying a chat template uses the tokenizer's default
        >>> config = {"name": "meta-llama/Llama-3.2-1B-Instruct"}
        >>> tokenizer = get_tokenizer(config)
        No chat template provided, using tokenizer's default
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful AI assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").apply_chat_template(messages, tokenize=False)

        >>> # Using a passthrough template
        >>> config = {
        ...     "name": "meta-llama/Llama-3.2-1B-Instruct",
        ...     "chat_template": None
        ... }
        >>> tokenizer = get_tokenizer(config)
        Using passthrough chat template
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == "".join(msg["content"] for msg in messages)

        >>> # Using a custom template
        >>> config = {
        ...     "name": "meta-llama/Llama-3.2-1B-Instruct",
        ...     "chat_template": "{% for message in messages %}{{ ' START: ' + message['content'] + ' END.' }}{% endfor %}"
        ... }
        >>> tokenizer = get_tokenizer(config)
        Using custom chat template
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == " START: You are a helpful AI assistant. END. START: Hello! END."
        ```
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "chat_template" in tokenizer_config:
        if tokenizer_config["chat_template"] is None:
            print("Using passthrough chat template")
            tokenizer.chat_template = (
                hf_datasets.COMMON_CHAT_TEMPLATES.passthrough_prompt_response
            )
        elif tokenizer_config["chat_template"].lower() == "default":
            print("Using tokenizer's default chat template")
        else:
            print("Using custom chat template")
            tokenizer.chat_template = tokenizer_config["chat_template"]
    else:
        print("No chat template provided, using tokenizer's default")

    return tokenizer


######## utils for main entry point functions ########


def setup_checkpointer(
    checkpointer_config: CheckpointingConfig,
    default_save_state: TypedDict,
) -> Tuple[CheckpointManager, Optional[str], TypedDict]:
    """Configure and initialize a checkpoint manager.

    Args:
        checkpointer_config (CheckpointingConfig): Configuration for checkpointing.
        default_save_state (TypedDict): Default state dictionary to use if no
            checkpoint exists. Should contain training progress tracking variables.

    Returns:
        tuple: A 3-tuple containing:
            - CheckpointManager: Initialized checkpoint manager
            - Optional[str]: Path to latest checkpoint if one exists, None otherwise
            - TypedDict: Training state dictionary, either loaded from checkpoint or default
    """
    checkpointer = CheckpointManager(checkpointer_config)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    if last_checkpoint_path is not None:
        save_state = checkpointer.load_training_info(last_checkpoint_path)
    else:
        save_state = default_save_state
    return checkpointer, last_checkpoint_path, save_state


def validate_checkpointing_config(
    checkpointer_config: CheckpointingConfig,
    algorithm_config: TypedDict,
) -> None:
    """Validate checkpointing configuration against algorithm configuration.

    Ensures that checkpointing is properly configured to save at intervals that align with
    validation periods, since validation metrics are used to determine which checkpoints to keep.

    Args:
        checkpointer_config (CheckpointingConfig): Checkpointing configuration.
        algorithm_config (TypedDict): Algorithm configuration.

    Raises:
        AssertionError: If checkpointing is enabled but save period is not valid, or if
            save period is not a multiple of validation period.
    """
    # config validation checks
    if checkpointer_config["enabled"]:
        assert checkpointer_config["save_period"] > 0
        assert (
            checkpointer_config["save_period"] % algorithm_config["val_period"] == 0
        ), (
            f"Checkpointing save period {checkpointer_config['save_period']} "
            f"must be a multiple of validation period {algorithm_config['val_period']}"
            f", or we won't know what metric to save!"
        )


def setup_dataloaders(
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
    collate_fn: Callable,
    algorithm_config: TypedDict,
    policy_config: PolicyConfig,
    last_checkpoint_path: Optional[str] = None,
    shuffle: bool = True,
) -> Tuple[StatefulDataLoader, Optional[StatefulDataLoader]]:
    """Setup training and validation dataloaders.

    Args:
        train_dataset (AllTaskProcessedDataset): Training dataset.
        val_dataset (Optional[AllTaskProcessedDataset]): Validation dataset.
        collate_fn (Callable): Collation function for dataset.
        algorithm_config (TypedDict): Algorithm configuration.
        policy_config (PolicyConfig): Policy configuration.
        last_checkpoint_path (Optional[str]): Path to latest checkpoint if one exists, None otherwise.
        shuffle (bool): Whether to shuffle the training dataloader.

    Returns:
        tuple: A 2-tuple containing (train_dataloader, val_dataloader).
    """
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(train_dataset)} samples")

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if algorithm_config["val_period"] > 0 or algorithm_config["val_at_start"]:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=(
                algorithm_config.get("val_global_batch_size", None)  ## sft, dpo
                or algorithm_config["val_batch_size"]  ## grpo
            ),
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

    return train_dataloader, val_dataloader


def save_checkpoint(
    checkpointer: CheckpointManager,
    master_config: TypedDict,
    save_state: TypedDict,
    total_steps: int,
    train_dataloader: StatefulDataLoader,
    policy: HfPolicy,
    timer: Timer,
) -> None:
    """Save a checkpoint.

    Args:
        checkpointer (CheckpointManager): The checkpoint manager to use for checkpointing
        master_config (TypedDict): The master configuration to dump to the checkpoint
        save_state (TypedDict): The save state to dump to the checkpoint
        total_steps (int): The number of training steps
        train_dataloader (StatefulDataLoader): The training dataloader
        policy (HfPolicy): The policy
        timer (Timer): The timer
    """
    with timer.time("checkpointing"):
        print(f"Saving checkpoint for step {total_steps}...")
        checkpoint_path = checkpointer.init_tmp_checkpoint(
            total_steps, save_state, master_config
        )
        policy.save_checkpoint(
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer"),
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
        )
        torch.save(
            train_dataloader.state_dict(),
            os.path.join(checkpoint_path, "train_dataloader.pt"),
        )
        checkpointer.finalize_checkpoint(checkpoint_path)


def reduce_microbatch_metrics(metrics: dict) -> dict:
    """Reduce microbatch metrics to a single value.

    For lr, global_valid_seqs, and global_valid_toks, takes the mean across microbatches.
    For all other metrics, takes the sum across microbatches.

    Args:
        metrics (dict): The metrics to reduce

    Returns:
        dict: The reduced metrics
    """
    for k, v in metrics.items():
        if k in {"lr", "global_valid_seqs", "global_valid_toks"}:
            metrics[k] = np.mean(v).item()
        else:
            metrics[k] = np.sum(v).item()
    return metrics


def log_metrics(
    log_to_console: dict,
    metrics: dict,
    timing_metrics: dict,
    step: int,
    logger: Logger,
    is_val: bool = False,
) -> None:
    """Log training or validation metrics both to console and to logger (wandb/tensorboard).

    Args:
        log_to_console (dict): Metrics to display in console output
        metrics (dict): Full metrics dictionary to log to logger
        timing_metrics (dict): Timing metrics to log to logger
        step (int): Current training step
        logger (Logger): Logger object
        is_val (bool, optional): Whether these are validation metrics. Defaults to False.

    The function:
    1. Prints metrics and timing information to console in a formatted way
    2. For training metrics, shows detailed timing breakdown
    3. Logs all metrics and timing info to the logger with appropriate prefixes
    """
    prefix: str = "validation" if is_val else "train"

    ## print metrics to std out
    print(f"\n📊 {prefix.capitalize()} Results:")
    for k, v in log_to_console.items():
        print(f"  • {k}: {v:.4f}")
    print(f"\n⏱️  {prefix.capitalize()} Timing:")

    total_time = (
        timing_metrics["total_step_time"]
        if prefix == "train"
        else timing_metrics["total_validation_time"]
    )
    print(f"  • Total {prefix} time: {total_time:.2f}s")

    if not is_val:
        # Display all other timing metrics (if any)
        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

    ## log metrics to wandb/tensorboard
    logger.log_metrics(metrics, step, prefix=f"{prefix}")
    logger.log_metrics(timing_metrics, step, prefix=f"timing/{prefix}")
