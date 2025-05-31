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
import math
import warnings
from collections import defaultdict
from functools import partial
from typing import TypedDict

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.loss_functions import (
    DPOLossFn,
)
from nemo_rl.algorithms.utils import (
    log_metrics,
    reduce_microbatch_metrics,
    save_checkpoint,
    set_seed,
    setup_checkpointer,
    setup_dataloaders,
    validate_checkpointing_config,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, dpo_collate_fn
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.hf_policy import HfPolicy
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.timer import Timer


class DPOConfig(TypedDict):
    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int

    reference_policy_kl_penalty: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool
    ## TODO(@ashors) support other loss functions
    ## https://github.com/NVIDIA/NeMo-RL/issues/193
    # preference_loss: str
    # gt_reward_scale: float
    preference_loss_weight: float
    sft_loss_weight: float


class DPOSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: float


def _default_dpo_save_state() -> DPOSaveState:
    return {
        "epoch": 1,
        "step": 1,
        "total_steps": 1,
    }


class DPOMasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    dpo: DPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: DPOMasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: AllTaskProcessedDataset,
) -> tuple[
    HfPolicy,
    RayVirtualCluster,
    StatefulDataLoader,
    StatefulDataLoader,
    DPOLossFn,
    Logger,
    CheckpointManager,
    DPOSaveState,
    DPOMasterConfig,
]:
    """Main entry point for running DPO algorithm.

    Returns:
        Tuple of policy, cluster, train_dataloader, val_dataloader, loss_fn, logger, checkpointer, dpo_save_state, master_config
    """
    set_seed(master_config["dpo"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    checkpointing_config = master_config["checkpointing"]
    dpo_config = master_config["dpo"]

    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    checkpointer, last_checkpoint_path, dpo_save_state = setup_checkpointer(
        checkpointing_config, _default_dpo_save_state()
    )

    # verify that checkpoint period is a multiple of validation period
    validate_checkpointing_config(checkpointing_config, dpo_config)

    # ==========================
    #           Data
    # ==========================
    ## TODO(@ashors) reduce boilerplate and move reused code into utils
    dpo_collate = partial(
        dpo_collate_fn,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=policy_config[
            "make_sequence_length_divisible_by"
        ],
    )
    train_dataloader, val_dataloader = setup_dataloaders(
        train_dataset,
        val_dataset,
        dpo_collate,
        dpo_config,
        policy_config,
        last_checkpoint_path,
    )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="dpo_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #   Training
    # ==========================
    policy = HfPolicy(cluster, policy_config, tokenizer, last_checkpoint_path)

    loss_fn = DPOLossFn(dpo_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        dpo_save_state,
        master_config,
    )


def add_ref_logprobs_to_data(dataloader, policy, master_config, is_val=False):
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)

            micro_batch_size = (
                master_config["dpo"]["val_micro_batch_size"] * 2
                if is_val
                else master_config["policy"]["train_micro_batch_size"] * 2
            )

            ## append ref policy logprobs to batch
            logprobs = policy.get_reference_policy_logprobs(
                batch,
                micro_batch_size=micro_batch_size,
            )["reference_logprobs"]
            ## want logprobs for batch to correspond to the log probabilities of the next tokens
            ## so we roll the logprobs to the left by one
            batch["reference_policy_logprobs"] = torch.roll(logprobs, -1, dims=-1)

            yield batch

        except StopIteration:
            break


def get_dpo_save_state(
    current_epoch, current_step, total_steps, val_metrics, train_dataloader
):
    dpo_save_state = {
        "step": (current_step + 1) % len(train_dataloader),
        "total_steps": total_steps + 1,
        "epoch": current_epoch,
        "val_loss": val_metrics["loss"],
    }
    return dpo_save_state


# =======================================================
# Training & Validation
# =======================================================
def validate(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    loss_fn,
    step: int,
    master_config: DPOMasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
):
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        val_metrics = defaultdict(lambda: 0.0)
        num_valid_batches = 0
        for batch_idx, val_batch in enumerate(
            add_ref_logprobs_to_data(val_dataloader, policy, master_config, is_val=True)
        ):
            ## just run model fwd
            val_results = policy.train(
                val_batch,
                loss_fn,
                eval_mode=True,
                gbs=val_batch_size * 2,
                mbs=val_mbs * 2,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )

            else:
                for k, v in val_results["all_mb_metrics"].items():
                    if k in {"lr", "global_valid_seqs", "global_valid_toks"}:
                        val_metrics[k] += np.mean(v).item()
                    else:
                        val_metrics[k] += np.sum(v).item()
                num_valid_batches += 1

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        for k, v in val_metrics.items():
            if k == "num_valid_samples":
                continue
            val_metrics[k] /= num_valid_batches

        # Calculate validation metrics
        policy.prepare_for_training()

    if len(val_metrics) == 0:
        warnings.warn(
            "No validation metrics were collected."
            " This is likely because there were no valid samples in the validation set."
        )
        log_to_console = {}

    else:
        log_to_console = {
            "loss": float(val_metrics["loss"]),
        }

    # Make sure to reset the timer after validation
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    timer.reset()

    return val_metrics, timing_metrics, log_to_console


def dpo_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    checkpointer,
    dpo_save_state,
):
    # Run dpo training
    timer = Timer()

    if dpo_save_state is None:
        dpo_save_state = _default_dpo_save_state()

    current_epoch = dpo_save_state["epoch"]
    current_step = dpo_save_state["step"]
    total_steps = dpo_save_state["total_steps"]

    dpo_config = master_config["dpo"]
    # Validation configuration
    val_period = dpo_config["val_period"]
    val_at_start = dpo_config["val_at_start"]
    max_num_epochs = dpo_config["max_num_epochs"]

    # Run validation at the start if configured
    if val_at_start and total_steps == 1:
        print("\n🔍 Running initial validation...")
        val_metrics, timing_metrics, log_to_console = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            val_batches=dpo_config["val_batches"],
            val_batch_size=dpo_config["val_global_batch_size"],
            val_mbs=dpo_config["val_micro_batch_size"],
        )
        log_metrics(
            log_to_console,
            val_metrics,
            timing_metrics,
            0,
            logger,
            is_val=True,
        )

    policy.prepare_for_training()

    total_num_epochs = min(
        max_num_epochs,
        math.ceil(master_config["dpo"]["max_num_steps"] / len(train_dataloader)),
    )

    while (
        current_epoch <= max_num_epochs
        and total_steps <= master_config["dpo"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch}/{total_num_epochs} {'=' * 25}")

        remaining_num_steps = master_config["dpo"]["max_num_steps"] % len(
            train_dataloader
        )
        num_steps_in_this_epoch = (
            remaining_num_steps
            if current_epoch == total_num_epochs
            else len(train_dataloader)
        )

        for batch in add_ref_logprobs_to_data(train_dataloader, policy, master_config):
            print(
                f"\n{'=' * 25} Step {current_step}/{num_steps_in_this_epoch} {'=' * 25}"
            )
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                print("▶ Taking a training step...")
                train_results = policy.train(
                    batch,
                    loss_fn,
                    eval_mode=False,
                    ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                    ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                    gbs=master_config["policy"]["train_global_batch_size"] * 2,
                    mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                )

                is_last_step = total_steps >= master_config["dpo"]["max_num_steps"] or (
                    current_epoch == max_num_epochs
                    and current_step == len(train_dataloader)
                )

                # Run validation if it's a validation step
                if is_last_step or (val_period > 0 and total_steps % val_period == 0):
                    val_metrics, timing_metrics, log_to_console = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps,
                        master_config=master_config,
                        val_batches=dpo_config["val_batches"],
                        val_batch_size=dpo_config["val_global_batch_size"],
                        val_mbs=dpo_config["val_micro_batch_size"],
                    )
                    log_metrics(
                        log_to_console,
                        val_metrics,
                        timing_metrics,
                        total_steps,
                        logger,
                        is_val=True,
                    )

                ## Checkpointing
                if master_config["checkpointing"]["enabled"] and (
                    is_last_step
                    or total_steps % master_config["checkpointing"]["save_period"] == 0
                ):
                    dpo_save_state = get_dpo_save_state(
                        current_epoch,
                        current_step,
                        total_steps,
                        val_metrics,
                        train_dataloader,
                    )
                    save_checkpoint(
                        checkpointer,
                        master_config,
                        dpo_save_state,
                        total_steps,
                        train_dataloader,
                        policy,
                        timer,
                    )

            losses = train_results["loss"]
            metrics = {
                "loss": train_results["loss"].numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
            }
            metrics.update(reduce_microbatch_metrics(train_results["all_mb_metrics"]))

            log_to_console = {
                "loss": float(metrics["loss"]),
            }
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")
            log_metrics(log_to_console, metrics, timing_metrics, total_steps, logger)

            timer.reset()
            current_step += 1
            total_steps += 1

            if total_steps > master_config["dpo"]["max_num_steps"]:
                return

        current_epoch += 1
        current_step = 1  # Reset step counter for new epoch
