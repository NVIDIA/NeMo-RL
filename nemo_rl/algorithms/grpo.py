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
from typing import Any, Optional, TypedDict, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    log_metrics,
    reduce_microbatch_metrics,
    save_checkpoint,
    setup_checkpointer,
    setup_dataloaders,
    validate_checkpointing_config,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import (
    DatumSpec,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.hf_policy import HfPolicy
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.timer import Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    max_rollout_turns: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    max_val_samples: int


class GRPOSaveState(TypedDict):
    step: int
    val_reward: float


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 1,
        "val_reward": -99999999.0,
    }


class GRPOLoggerConfig(LoggerConfig):
    num_val_samples_to_print: int  # number of val samples to print to stdout


class GRPOMasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: GRPOMasterConfig,
    tokenizer: TokenizerType,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    GRPOSaveState,
    GRPOMasterConfig,
]:
    """Main entry point for running GRPO algorithm.

    Returns:
        Tuple of policy, cluster, train_dataloader, val_dataloader, loss_fn, logger, checkpointer, grpo_save_state, master_config
    """
    policy_config = master_config["policy"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    checkpointing_config = master_config["checkpointing"]
    loss_config = master_config["loss_fn"]
    grpo_config = master_config["grpo"]
    generation_config = policy_config["generation"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for GRPO"
    )

    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    checkpointer, last_checkpoint_path, grpo_save_state = setup_checkpointer(
        checkpointing_config, _default_grpo_save_state()
    )

    # verify that checkpoint period is a multiple of validation period
    validate_checkpointing_config(checkpointing_config, grpo_config)

    train_dataloader, val_dataloader = setup_dataloaders(
        dataset,
        val_dataset,
        rl_collate_fn,
        grpo_config,
        policy_config,
        last_checkpoint_path,
        shuffle=False,
    )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    colocated_inference = generation_config["backend"] != "hf"
    cluster = RayVirtualCluster(
        name="grpo_policy_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=2 if colocated_inference else 1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...")

    # vllm model loading prefers clean environment, initialize policy_generation before policy (#52 will fix this)
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "hf":
        policy_generation = None
        print(f"  ✓ Using HF backend for generation with {policy_config['model_name']}")
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        policy_generation = VllmGeneration(cluster=cluster, config=generation_config)
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        policy_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    policy = HfPolicy(cluster, policy_config, tokenizer, last_checkpoint_path)

    # initialize loss function
    loss_fn = ClippedPGLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        policy_generation,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
        master_config,
    )


def get_grpo_save_state(step, val_metrics):
    grpo_save_state = {
        "step": step + 1,
        "val_reward": val_metrics["accuracy"],
    }
    return grpo_save_state


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================


def refit_policy_generation(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    refit_buffer_size_gb: int,  # GB
) -> None:
    """Refit the policy generation interface with the latest policy weights."""
    policy.offload_before_refit()
    policy_generation.prepare_for_generation(tags=["weights"])
    # Streaming update weights to save memory
    state_dict_info: list[tuple[str, int]] = policy.prepare_weights_for_ipc()
    # group keys to save time
    available_bytes = refit_buffer_size_gb * (1024**3)
    split_keys: list[list[str]] = []
    keys: list[str] = []
    for key, size_in_bytes in state_dict_info:
        if size_in_bytes > available_bytes:
            if keys:
                split_keys.append(keys)
                keys = []
            available_bytes = refit_buffer_size_gb * (1024**3)

        keys.append(key)
        available_bytes -= size_in_bytes

    if len(keys) > 0:
        split_keys.append(keys)
    # do update
    for keys in split_keys:
        ipc_handles = policy.get_weights_ipc_handles(keys)
        if not policy_generation.update_weights(ipc_handles):
            error_message = (
                "❌ Error: Updating weights for the generation policy failed during refit.\n"
                "This often indicates an issue with cuda-ipc or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)
    policy.offload_after_refit()
    policy_generation.prepare_for_generation(tags=["kv_cache"])


# ===============================================================================
# Training & Validation
# ===============================================================================


def grpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: GRPOMasterConfig,
) -> None:
    """Run GRPO training algorithm."""
    timer = Timer()
    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (hf framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert policy_generation is not None  # for mypy type check

    # common config/state itmes
    step = grpo_save_state["step"]
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    refit_buffer_size_gb = master_config["policy"]["refit_buffer_size_gb"]

    # Run validation at the start if configured
    if val_at_start and step == 1:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, refit_buffer_size_gb)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, timing_metrics, log_to_console = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
        )
        policy_generation.finish_generation()

        log_metrics(
            log_to_console,
            val_metrics,
            timing_metrics,
            0,
            logger,
            is_val=True,
        )

    # Run grpo training (single-turn)
    batch: BatchedDataDict[DatumSpec]
    for batch in dataloader:
        print(
            f"\n{'=' * 25} Step {step}/{min(len(dataloader), master_config['grpo']['max_num_steps'])} {'=' * 25}"
        )
        val_metrics, validation_timings = None, None

        with timer.time("total_step_time"):
            # Prepare batch
            print("▶ Preparing batch...")
            with timer.time("data_processing"):
                # Repeat batch items
                repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                    master_config["grpo"]["num_generations_per_prompt"]
                )
                # Convert LLMMessageLogType to FlatMessagesType for generation
                batched_flat, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                input_ids = batched_flat["token_ids"]

            # Generate responses - this updates the LLMMessageLogType in repeated_batch
            print(f"▶ Generating responses for batch of size {repeated_batch.size}...")
            with timer.time("prepare_for_generation"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size_gb,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            with timer.time("generation"):
                repeated_batch, rollout_metrics = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )
                policy_generation.finish_generation()

            # Calculate rewards & advantages
            print("▶ Processing rewards...")
            with timer.time("reward_calculation"):
                # Extract rewards from final_batch
                rewards = repeated_batch["total_reward"]

                print("▶ Computing advantages...")
                baseline, std = calculate_baseline_and_std_per_prompt(
                    input_ids,
                    rewards,
                    torch.ones_like(rewards),
                    leave_one_out_baseline=master_config["grpo"][
                        "use_leave_one_out_baseline"
                    ],
                )
                advantages = (rewards - baseline).unsqueeze(-1)

                if master_config["grpo"]["normalize_rewards"]:
                    # don't sharpen the ones with no variation
                    zero_std_mask = std > 0
                    advantages[zero_std_mask] = (
                        advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
                    )

            with timer.time("data_processing"):
                # Add loss mask and advantages to each message in LLMMessageLogType
                for i, message_log in enumerate(repeated_batch["message_log"]):
                    for j, message in enumerate(message_log):
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(
                                message["token_ids"]
                            )
                        else:
                            message["token_loss_mask"] = torch.zeros_like(
                                message["token_ids"]
                            )
                        if "generation_logprobs" not in message:
                            message["generation_logprobs"] = torch.zeros_like(
                                message["token_ids"], dtype=torch.float32
                            )
                        message["advantages"] = advantages[i].expand(
                            message["token_ids"].shape
                        )

                # Convert updated LLMMessageLogType to FlatMessagesType for training
                flat_messages, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"][
                        "make_sequence_length_divisible_by"
                    ],
                )

                # Create training data from flattened messages
                train_data = BatchedDataDict[ClippedPGLossDataDict](
                    {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "advantages": flat_messages["advantages"],
                        "generation_logprobs": flat_messages["generation_logprobs"],
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                )
                train_data.to("cpu")

            print("▶ Preparing for logprob inference...")
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            print("▶ Computing logprobs...")
            with timer.time("policy_and_reference_logprobs"):
                fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(train_data)[
                    "reference_logprobs"
                ]
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

            print("▶ Preparing for training...")
            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            print("▶ Training policy...")
            with timer.time("policy_training"):
                train_results = policy.train(train_data, loss_fn)

            is_last_step = step == min(
                master_config["grpo"]["max_num_steps"], len(dataloader)
            )

            # Run validation if it's a validation step
            if is_last_step or (val_period > 0 and step % val_period == 0):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size_gb,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
                val_metrics, timing_metrics, log_to_console = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step,
                    master_config=master_config,
                )
                policy_generation.finish_generation()

                log_metrics(
                    log_to_console,
                    val_metrics,
                    timing_metrics,
                    step,
                    logger,
                    is_val=True,
                )

            ## Checkpointing
            if master_config["checkpointing"]["enabled"] and (
                is_last_step
                or step % master_config["checkpointing"]["save_period"] == 0
            ):
                policy.prepare_for_training()

                grpo_save_state = get_grpo_save_state(step, val_metrics)
                save_checkpoint(
                    checkpointer,
                    master_config,
                    grpo_save_state,
                    step,
                    dataloader,
                    policy,
                    timer,
                )
                policy.offload_after_refit()

        # Logging
        # Log training data
        log_data = {
            "content": flat_messages["content"],
            "rewards": rewards.tolist(),
            "generation_logprobs": train_data["generation_logprobs"].tolist(),
            "prev_logprobs": train_data["prev_logprobs"].tolist(),
            "input_lengths": input_lengths.tolist(),
        }
        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

        metrics = {
            "loss": train_results["loss"].numpy(),
            "reward": np.mean(rewards.numpy()),
            "grad_norm": train_results["grad_norm"].numpy(),
        }
        metrics.update(reduce_microbatch_metrics(train_results["all_mb_metrics"]))
        metrics.update(rollout_metrics)

        log_to_console = {
            "loss": metrics["loss"],
            "Avg Reward": np.mean(rewards.numpy()),
            "Avg Generation Length": rollout_metrics["mean_gen_tokens_per_sample"],
        }
        timing_metrics: dict[str, float] = timer.get_timing_metrics(reduction_op="sum")  # type: ignore
        log_metrics(log_to_console, metrics, timing_metrics, step, logger, is_val=False)

        timer.reset()
        step += 1
        if step > master_config["grpo"]["max_num_steps"]:
            break


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: GRPOMasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        total_rewards = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config["grpo"]["max_val_samples"]
            // master_config["grpo"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            val_batch, gen_metrics = run_multi_turn_rollout(
                policy_generation,
                val_batch,
                tokenizer,
                val_task_to_env,
                max_seq_len=master_config["policy"]["max_total_sequence_length"],
                max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                greedy=False,
            )
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = get_keys_from_message_log(
                val_batch["message_log"], ["role", "content"]
            )
            all_message_logs.append(to_env)

        # Calculate validation metrics
        accuracy = sum(total_rewards) / len(total_rewards)
        avg_length = sum(total_lengths) / len(total_lengths)

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...")

    # Get timing metrics
    log_to_console = {
        "Accuracy": accuracy,
        "Average response length:": avg_length,
        "Samples processed:": len(total_rewards),
    }

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics, log_to_console
