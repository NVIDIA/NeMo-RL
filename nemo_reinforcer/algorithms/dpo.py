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
from functools import partial
from typing import Optional, Tuple, TypedDict
from tqdm import tqdm

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from nemo_reinforcer.algorithms.loss_functions import (
    DPOLossFn,
)
from nemo_reinforcer.algorithms.utils import set_seed, get_tokenizer
from nemo_reinforcer.data import DataConfig
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset, dpo_collate_fn
from nemo_reinforcer.data.interfaces import TaskDataSpec
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_reinforcer.models.interfaces import PolicyInterface
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.utils.checkpoint import CheckpointManager, CheckpointingConfig
from nemo_reinforcer.utils.logger import Logger, LoggerConfig
from nemo_reinforcer.utils.timer import Timer


class DPOSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: float
    consumed_samples: int


def _default_dpo_save_state() -> DPOSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


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
    ## https://github.com/NVIDIA/reinforcer/issues/193
    # preference_loss: str
    # gt_reward_scale: float
    preference_loss_weight: float
    sft_loss_weight: float


class MasterConfig(TypedDict):
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
    master_config: MasterConfig,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: AllTaskProcessedDataset,
) -> Tuple[
    HfPolicy,
    RayVirtualCluster,
    StatefulDataLoader,
    StatefulDataLoader,
    DPOLossFn,
    MasterConfig,
    Logger,
    TaskDataSpec,
    DPOSaveState,
]:
    """Main entry point for running DPO algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    set_seed(master_config["dpo"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    dpo_config = master_config["dpo"]

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    dpo_save_state: Optional[DPOSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )
    # config validation checks
    if master_config["checkpointing"]["enabled"]:
        assert master_config["checkpointing"]["save_period"] > 0
        assert (
            master_config["checkpointing"]["save_period"]
            % master_config["dpo"]["val_period"]
            == 0
        ), (
            f"Checkpointing save period {master_config['checkpointing']['save_period']} "
            f"must be a multiple of validation period {master_config['dpo']['val_period']}"
            f", or we won't know what metric to save!"
        )

    # ==========================
    #           Data
    # ==========================
    ## TODO: clean up
    tokenizer = get_tokenizer(policy_config["model_name"])
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=True,
        collate_fn=partial(dpo_collate_fn, tokenizer=tokenizer),
        drop_last=True,
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataloader = StatefulDataLoader(
        val_dataset,
        batch_size=dpo_config["val_global_batch_size"],
        shuffle=False,
        collate_fn=partial(dpo_collate_fn, tokenizer=tokenizer),
        drop_last=True,
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
    print("\n▶ Setting up model...")
    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
        init_reference_model=True,
    )
    loss_fn = DPOLossFn(master_config["dpo"])
    print(f"  ✓ Model initialized")

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


def augment_dataloader(dataloader, policy, master_config):
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)

            ## append ref policy logprobs to batch
            logprobs = policy.get_reference_policy_logprobs(
                batch,
                micro_batch_size=master_config["policy"]["train_micro_batch_size"] * 2,
            )["reference_logprobs"].to("cpu")
            ## want logprobs for batch to correspond to the log probabilities of the next tokens
            ## so we roll the logprobs to the left by one
            batch["reference_policy_logprobs"] = torch.roll(logprobs, -1, dims=-1)

            yield batch

        except StopIteration:
            break


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

        for batch_idx, val_batch in enumerate(
            augment_dataloader(val_dataloader, policy, master_config)
        ):
            ## just run model fwd
            val_results = policy.train(
                val_batch,
                loss_fn,
                eval_mode=True,
                gbs=val_batch_size * 2,
                mbs=val_mbs * 2,
            )

            val_metrics = {
                "loss": val_results["loss"].numpy(),
            }
            val_metrics.update(val_results["all_mb_metrics"])
            val_metrics = {k: np.mean(v).item() for k, v in val_metrics.items()}
            if val_batches > 0 and batch_idx >= val_batches:
                break

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Validation loss: {float(val_metrics['loss']):.4f}")
    print(f"    • Validation accuracy: {float(val_metrics['accuracy']):.4f}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics


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
        current_epoch = 0
        current_step = 0
        total_steps = 0
    else:
        current_epoch = dpo_save_state["epoch"]
        current_step = dpo_save_state["step"]
        total_steps = dpo_save_state["total_steps"]

    dpo_config = master_config["dpo"]
    # Validation configuration
    val_period = dpo_config["val_period"]
    val_at_start = dpo_config["val_at_start"]
    max_num_epochs = dpo_config["max_num_epochs"]

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...")
        val_metrics, validation_timings = validate(
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

        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    policy.prepare_for_training()

    while current_epoch < max_num_epochs:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in augment_dataloader(train_dataloader, policy, master_config):
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['dpo']['max_num_steps'])} {'=' * 25}"
            )

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

                # Run validation if it's a validation step
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    val_metrics, validation_timings = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        val_batches=dpo_config["val_batches"],
                        val_batch_size=dpo_config["val_global_batch_size"],
                        val_mbs=dpo_config["val_micro_batch_size"],
                    )
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                ## Checkpointing
                dpo_save_state["consumed_samples"] += master_config["policy"][
                    "train_global_batch_size"
                ]
                if (
                    master_config["checkpointing"]["enabled"]
                    and (total_steps + 1)
                    % master_config["checkpointing"]["save_period"]
                    == 0
                ):  # +1 because step is 0-indexed
                    is_last_checkpoint = (
                        min(
                            len(train_dataloader) * max_num_epochs,
                            master_config["dpo"]["max_num_steps"],
                        )
                        - (total_steps + 1)
                        < master_config["checkpointing"]["save_period"]
                    )
                    dpo_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    dpo_save_state["total_steps"] = total_steps + 1
                    dpo_save_state["epoch"] = current_epoch
                    dpo_save_state["val_loss"] = val_metrics["loss"]
                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, dpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            save_hf=is_last_checkpoint,
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            losses = train_results["loss"]
            metrics = {
                "loss": train_results["loss"].numpy(),
            }
            metrics.update(train_results["all_mb_metrics"])
            metrics = {k: np.mean(v).item() for k, v in metrics.items()}
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results:")
            print(f"  • Loss: {float(metrics['loss']):.4f}")
            print("\n⏱️  Timing:")
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")

            # Display all other timing metrics (if any)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if total_steps >= master_config["dpo"]["max_num_steps"]:
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
