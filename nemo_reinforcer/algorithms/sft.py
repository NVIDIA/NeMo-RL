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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypedDict

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from nemo_reinforcer.algorithms.loss_functions import (
    NLLLoss,
)
from nemo_reinforcer.data import DataConfig, hf_datasets
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_reinforcer.data.interfaces import TaskDataSpec, DatumSpec
from nemo_reinforcer.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
    get_formatted_message_log,
)
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_reinforcer.models.interfaces import PolicyInterface
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.utils.checkpoint import CheckpointManager, CheckpointingConfig
from nemo_reinforcer.utils.logger import Logger, LoggerConfig
from nemo_reinforcer.utils.timer import Timer


class SFTSaveState(TypedDict):
    step: int
    val_loss: float
    consumed_samples: int


def _default_sft_save_state() -> SFTSaveState:
    return {
        "step": 0,
        "consumed_samples": 0,
    }


class SFTConfig(TypedDict):
    num_steps: int
    val_period: int
    val_at_start: bool
    checkpoint_dir: str


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# =======================================================
# Data Processing
# =======================================================
def sft_preprocessor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for SFT training."""
    message_log = get_formatted_message_log(
        datum_dict["messages"], tokenizer, task_data_spec
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
) -> Tuple[
    HfPolicy,
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    AutoTokenizer,
    NLLLoss,
    MasterConfig,
    Logger,
    TaskDataSpec,
    SFTSaveState,
]:
    """Main entry point for running SFT algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    sft_config = master_config["sft"]

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    sft_save_state: Optional[SFTSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )
    # config validation checks
    if master_config["checkpointing"]["enabled"]:
        assert master_config["checkpointing"]["save_period"] > 0
        assert (
            master_config["checkpointing"]["save_period"]
            % master_config["sft"]["val_period"]
            == 0
        ), (
            f"Checkpointing save period {master_config['checkpointing']['save_period']} "
            f"must be a multiple of validation period {master_config['sft']['val_period']}"
            f", or we won't know what metric to save!"
        )

    # ==========================
    #           Data
    # ==========================
    print("\n▶ Setting up data...")
    data_cls = data_config["dataset_name"]
    if data_cls == "open_assistant":
        data = hf_datasets.OasstDataset(output_dir="/tmp/open_assistant")
    elif data_cls == "squad":
        data = hf_datasets.SquadDataset()
    else:
        raise ValueError(f"Unknown dataset class: {data_cls}")
    print(
        f"  ✓ Training and validation dataset loaded with {len(data.formatted_ds['train'])} and {len(data.formatted_ds['validation'])} samples, respectively."
    )

    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]
    sft_task_spec = data.task_spec

    tokenizer = AutoTokenizer.from_pretrained(policy_config["model_name"])

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        sft_task_spec,
        sft_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=True,
        collate_fn=rl_collate_fn,  ## TODO: change this for sft! or make it more general
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataset = AllTaskProcessedDataset(
        val_dataset,
        tokenizer,
        sft_task_spec,
        sft_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    ## TODO: support different batch sizes for train and val
    val_dataloader = StatefulDataLoader(
        val_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        drop_last=True,
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
    print("\n▶ Setting up model...")
    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        weights_path=Path(last_checkpoint_path) / "policy.pt"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy_optimizer.pt"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
    )
    loss_fn = NLLLoss()
    print(f"  ✓ Model initialized")

    logger = Logger(logger_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        sft_task_spec,
        checkpointer,
        sft_save_state,
    )


# =======================================================
# Training & Validation
# =======================================================
def validate(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    sft_task_spec: TaskDataSpec,
    val_batches: int,
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

        for batch_idx, val_batch in enumerate(val_dataloader):
            ## add loss mask based on role to every message
            add_loss_mask_to_message_log(
                val_batch["message_log"],
                roles_to_train_on=["assistant"],
            )

            cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                val_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.eos_token_id},
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
            val_results = policy.train(val_data, loss_fn, eval_mode=True)
            val_metrics["val_loss"] += float(val_results["loss"])

            if val_batches > 0 and batch_idx >= val_batches:
                break

        val_metrics["val_loss"] /= val_batches

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Validation loss: {val_metrics['val_loss']:.4f}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics


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

    if sft_save_state is None:
        sft_save_state = _default_sft_save_state()
        step = 0
    else:
        step = (
            sft_save_state["step"] + 1
        )  # N+1 because the checkpoint is _after_ SFT iteration N

    # Validation configuration
    val_period = master_config["sft"]["val_period"]
    val_at_start = master_config["sft"]["val_at_start"]

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        val_metrics, validation_timings = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            sft_task_spec=sft_task_spec,
            val_batches=master_config["sft"]["val_batches"],
        )

        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    policy.prepare_for_training()

    for batch in train_dataloader:
        print(f"\n{'=' * 25} Step {step + 1}/{len(train_dataloader)} {'=' * 25}")

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
                    pad_value_dict={"token_ids": tokenizer.eos_token_id},
                )

                train_data: BatchedDataDict = BatchedDataDict(
                    {
                        "input_ids": cat_and_padded["token_ids"],
                        "input_lengths": input_lengths,
                        "token_mask": cat_and_padded["token_loss_mask"],
                        "sample_mask": batch["loss_multiplier"],
                    }
                )

            ## train_data.to("cpu")
            print("▶ Taking a training step...")
            train_results = policy.train(train_data, loss_fn)

            # Run validation if it's a validation step
            if val_period > 0 and (step + 1) % val_period == 0:
                val_metrics, validation_timings = validate(
                    policy,
                    val_dataloader,
                    tokenizer,
                    loss_fn,
                    step=step + 1,
                    master_config=master_config,
                    sft_task_spec=sft_task_spec,
                    val_batches=master_config["sft"]["val_batches"],
                )
                logger.log_metrics(
                    validation_timings, step + 1, prefix="timing/validation"
                )
                logger.log_metrics(val_metrics, step + 1, prefix="validation")

            ## Checkpointing
            sft_save_state["consumed_samples"] += master_config["policy"][
                "train_global_batch_size"
            ]
            if (
                master_config["checkpointing"]["enabled"]
                and (step + 1) % master_config["checkpointing"]["save_period"] == 0
            ):  # +1 because step is 0-indexed
                sft_save_state["step"] = step
                sft_save_state["val_loss"] = val_metrics["val_loss"]
                with timer.time("checkpointing"):
                    print(f"Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        step + 1, sft_save_state, master_config
                    )
                    policy.save_checkpoint(
                        os.path.join(checkpoint_path, "policy.pt"),
                        os.path.join(checkpoint_path, "policy_optimizer.pt"),
                        offload_to_cpu=False,
                    )
                    torch.save(
                        train_dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)

        losses = train_results["loss"]
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        metrics = {
            "loss": losses.numpy(),
        }

        print("\n📊 Training Results:")
        print(f"  • Loss: {float(metrics['loss']):.4f}")
        print("\n⏱️  Timing:")
        # Display total time first, separately
        total_time = timing_metrics.get("total_step_time", 0)
        print(f"  • Total step time: {total_time:.2f}s")

        # Display all other timing metrics
        ## TODO: remove?
        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

        logger.log_metrics(metrics, step + 1, prefix="train")
        logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        timer.reset()
        step += 1

        if step >= master_config["sft"]["num_steps"] - 1:
            break
