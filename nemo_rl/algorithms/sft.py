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
from typing import TypedDict

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.loss_functions import (
    NLLLoss,
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
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.hf_policy import HfPolicy
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.timer import Timer


class SFTConfig(TypedDict):
    max_num_steps: int
    max_num_epochs: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int


class SFTSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: float


def _default_sft_save_state() -> SFTSaveState:
    return {
        "epoch": 1,
        "step": 1,
        "total_steps": 1,
    }


class SFTMasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: SFTMasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: AllTaskProcessedDataset,
) -> tuple[
    HfPolicy,
    RayVirtualCluster,
    StatefulDataLoader,
    StatefulDataLoader,
    NLLLoss,
    Logger,
    CheckpointManager,
    SFTSaveState,
    SFTMasterConfig,
]:
    """Main entry point for running SFT algorithm.

    Returns:
        Tuple of policy, cluster, train_dataloader, val_dataloader, loss_fn, logger, checkpointer, sft_save_state, master_config
    """
    set_seed(master_config["sft"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    checkpointing_config = master_config["checkpointing"]
    sft_config = master_config["sft"]

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer, last_checkpoint_path, sft_save_state = setup_checkpointer(
        checkpointing_config, _default_sft_save_state()
    )

    # verify that checkpoint period is a multiple of validation period
    validate_checkpointing_config(checkpointing_config, sft_config)

    # ==========================
    #           Data
    # ==========================
    train_dataloader, val_dataloader = setup_dataloaders(
        train_dataset,
        val_dataset,
        rl_collate_fn,
        sft_config,
        policy_config,
        last_checkpoint_path,
    )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="sft_cluster",
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
    policy = HfPolicy(
        cluster,
        policy_config,
        tokenizer,
        last_checkpoint_path,
        init_reference_model=False,
    )

    # initialize loss function
    loss_fn = NLLLoss()

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
        sft_save_state,
        master_config,
    )


def get_sft_save_state(
    current_epoch, current_step, total_steps, val_metrics, train_dataloader
):
    sft_save_state = {
        "step": (current_step + 1) % len(train_dataloader),
        "total_steps": total_steps + 1,
        "epoch": current_epoch,
        "val_loss": val_metrics["val_loss"],
    }
    return sft_save_state


# =======================================================
# Training & Validation
# =======================================================
def validate(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    loss_fn,
    step: int,
    master_config: SFTMasterConfig,
    sft_task_spec: TaskDataSpec,
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

        # Show a progress indicator for validation
        # val_total = len(val_dataloader)

        val_metrics = {"val_loss": 0.0}
        num_valid_batches = 0

        policy.prepare_for_training()
        for batch_idx, val_batch in enumerate(val_dataloader):
            ## add loss mask based on role to every message
            add_loss_mask_to_message_log(
                val_batch["message_log"],
                roles_to_train_on=["assistant"],
            )

            cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                val_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
                make_sequence_length_divisible_by=master_config["policy"][
                    "make_sequence_length_divisible_by"
                ],
            )

            val_data: BatchedDataDict = BatchedDataDict(
                {
                    "input_ids": cat_and_padded["token_ids"],
                    "input_lengths": input_lengths,
                    "token_mask": cat_and_padded["token_loss_mask"],
                    "sample_mask": val_batch["loss_multiplier"],
                }
            )

            ## just run model fwd
            val_results = policy.train(
                val_data,
                loss_fn,
                eval_mode=True,
                gbs=val_batch_size,
                mbs=val_mbs,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                val_metrics["val_loss"] += float(val_results["loss"])
                num_valid_batches += 1

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if num_valid_batches > 0:
            val_metrics["val_loss"] /= num_valid_batches
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )

        # Calculate validation metrics
        policy.prepare_for_training()

    if num_valid_batches > 0:
        log_to_console = {
            "loss": val_metrics["val_loss"],
        }
    else:
        log_to_console = {}

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics, log_to_console


def sft_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    sft_task_spec,
    checkpointer,
    sft_save_state,
):
    # Run basic sft training
    timer = Timer()

    current_epoch = sft_save_state["epoch"]
    current_step = sft_save_state["step"]
    total_steps = sft_save_state["total_steps"]

    sft_config = master_config["sft"]
    # Validation configuration
    val_period = sft_config["val_period"]
    val_at_start = sft_config["val_at_start"]
    max_num_epochs = sft_config["max_num_epochs"]

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
            sft_task_spec=sft_task_spec,
            val_batches=sft_config["val_batches"],
            val_batch_size=sft_config["val_global_batch_size"],
            val_mbs=sft_config["val_micro_batch_size"],
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
        math.ceil(master_config["sft"]["max_num_steps"] / len(train_dataloader)),
    )

    while (
        current_epoch <= total_num_epochs
        and total_steps <= master_config["sft"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch}/{total_num_epochs} {'=' * 25}")

        remaining_num_steps = master_config["sft"]["max_num_steps"] % len(
            train_dataloader
        )
        num_steps_in_this_epoch = (
            remaining_num_steps
            if current_epoch == total_num_epochs
            else len(train_dataloader)
        )

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step}/{num_steps_in_this_epoch} {'=' * 25}"
            )
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch and generate responses
                print("▶ Preparing batch...")
                with timer.time("data_processing"):
                    ## add loss mask based on role to every message
                    add_loss_mask_to_message_log(
                        batch["message_log"],
                        roles_to_train_on=["assistant"],
                    )

                    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                        batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data: BatchedDataDict = BatchedDataDict(
                        {
                            "input_ids": cat_and_padded["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": cat_and_padded["token_loss_mask"],
                            "sample_mask": batch["loss_multiplier"],
                        }
                    )

                print("▶ Taking a training step...")
                train_results = policy.train(train_data, loss_fn)

                is_last_step = total_steps >= master_config["sft"]["max_num_steps"] or (
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
                        sft_task_spec=sft_task_spec,
                        val_batches=sft_config["val_batches"],
                        val_batch_size=sft_config["val_global_batch_size"],
                        val_mbs=sft_config["val_micro_batch_size"],
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
                    sft_save_state = get_sft_save_state(
                        current_epoch,
                        current_step,
                        total_steps,
                        val_metrics,
                        train_dataloader,
                    )
                    save_checkpoint(
                        checkpointer,
                        master_config,
                        sft_save_state,
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
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            log_to_console = {
                "loss": float(metrics["loss"]),
            }
            log_metrics(
                log_to_console,
                metrics,
                timing_metrics,
                total_steps,
                logger,
                is_val=False,
            )

            timer.reset()
            current_step += 1
            total_steps += 1

            if total_steps > master_config["sft"]["max_num_steps"]:
                return

        current_epoch += 1
        current_step = 1  # Reset step counter for new epoch
