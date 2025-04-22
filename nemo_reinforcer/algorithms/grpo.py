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
from typing import Any, Dict, Tuple, TypedDict, Iterable, Optional, List

import os
from pathlib import Path
import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.algorithms.utils import calculate_baseline_and_std_per_prompt

from nemo_reinforcer.environments.interfaces import EnvironmentInterface
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    FlatMessagesType,
)
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.models.generation.vllm import VllmGeneration
from nemo_reinforcer.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.data import DataConfig
from nemo_reinforcer.data.llm_message_utils import (
    get_keys_from_message_log,
    batched_message_log_to_flat_message,
)
from nemo_reinforcer.utils.logger import (
    print_message_log_samples,
)
from nemo_reinforcer.distributed.virtual_cluster import ClusterConfig
from nemo_reinforcer.environments.math_environment import MathEnvConfig
from nemo_reinforcer.models.generation.interfaces import (
    GenerationInterface,
    GenerationDatumSpec,
)
from nemo_reinforcer.models.interfaces import PolicyInterface
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.utils.logger import Logger, LoggerConfig
from nemo_reinforcer.utils.timer import Timer
from nemo_reinforcer.utils.checkpoint import CheckpointManager, CheckpointingConfig


# ===============================================================================
# Configuration
# ===============================================================================


class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_at_start: bool
    checkpoint_dir: str


class GRPOSaveState(TypedDict):
    step: int
    val_reward: float
    consumed_samples: int


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
    }


class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    math_env: MathEnvConfig
    data: DataConfig
    grpo: GRPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> Tuple[
    PolicyInterface,
    GenerationInterface,
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    GRPOSaveState,
    MasterConfig,
]:
    """Main entry point for running GRPO algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, logger, master_config, val_dataloader
    """
    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    data_config = master_config["data"]
    grpo_config = master_config["grpo"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

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
    grpo_save_state: Optional[GRPOSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )
    if grpo_save_state is None:
        grpo_save_state = _default_grpo_save_state()

    # config validation checks
    if master_config["checkpointing"]["enabled"]:
        assert master_config["checkpointing"]["save_period"] > 0
        assert (
            master_config["checkpointing"]["save_period"]
            % master_config["grpo"]["val_period"]
            == 0
        ), (
            f"Checkpointing save period {master_config['checkpointing']['save_period']} "
            f"must be a multiple of validation period {master_config['grpo']['val_period']}"
            f", or we won't know what metric to save!"
        )

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=rl_collate_fn,
    )
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

    # Load validation dataset if provided
    val_dataloader = None
    # If validation is enabled, load the validation dataloader
    if grpo_config["val_period"] > 0 or grpo_config["val_at_start"]:
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=grpo_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

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
        policy_generation = VllmGeneration(cluster=cluster, config=generation_config)
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        policy_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
    )

    loss_fn = ClippedPGLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
        master_config,
    )


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================


def refit_policy_generation(
    policy: PolicyInterface,
    policy_generation: GenerationInterface,
    refit_buffer_size: int,  # GB
):
    """Refit the policy generation interface with the latest policy weights."""
    policy.offload_before_refit()
    policy_generation.prepare_for_generation(tags=["weights"])
    # Streaming update weights to save memory
    state_dict_info = policy.prepare_weights_for_ipc()
    # group keys to save time
    available_bytes = refit_buffer_size * (1024**3)
    split_keys, keys = [], []
    for key, size_in_bytes in state_dict_info:
        keys.append(key)
        available_bytes -= size_in_bytes
        if available_bytes <= 0:
            split_keys.append(keys)
            keys = []
            available_bytes = refit_buffer_size * (1024**3)
    if len(keys) > 0:
        split_keys.append(keys)
    # do update
    for keys in split_keys:
        ipc_handles = policy.get_weights_ipc_handles(keys)
        policy_generation.update_weights(ipc_handles)
    policy.offload_after_refit()
    policy_generation.prepare_for_generation(tags=["kv_cache"])


def generate_responses(
    policy_generation: GenerationInterface,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[DatumSpec],
    tokenizer,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
) -> Tuple[BatchedDataDict[DatumSpec], List[List[int]], Dict[str, float | int]]:
    """Generate responses from policy."""
    # Generate responses
    generation_outputs = policy_generation.generate(generation_input_data)

    # Extract generated tokens
    generated_ids = []
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]
    for output_ids, input_length, total_length in zip(
        generation_outputs["output_ids"], input_lengths, unpadded_sequence_lengths
    ):
        generated_ids.append(output_ids[input_length:total_length])

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Append to message log
    for i, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        message = {
            "role": "assistant",
            "content": text,
            "token_ids": generation_outputs["output_ids"][i, input_length:total_length],
        }

        if include_logprobs and "logprobs" in generation_outputs:
            message["generation_logprobs"] = generation_outputs["logprobs"][
                i, input_length:total_length
            ]

        batch["message_log"][i].append(message)

    metrics = {
        "mean_generation_length": (
            torch.sum(unpadded_sequence_lengths) - torch.sum(input_lengths)
        ).item()
        / len(unpadded_sequence_lengths),
        "max_seqlen": torch.max(unpadded_sequence_lengths).item(),
    }

    return batch, generated_ids, metrics


def calculate_rewards(
    batch: BatchedDataDict[DatumSpec],
    task_to_env: Dict[str, EnvironmentInterface],
) -> Tuple[torch.Tensor, List[LLMMessageLogType]]:
    """Calculate rewards for generated responses.

    Args:
        batch: Batch containing message_log (LLMMessageLogType) with generated responses
        task_to_env: Dictionary mapping task names to their corresponding environments

    Returns:
        rewards: Tensor of rewards
        to_env: Simplified message logs sent to environment (LLMMessageLogType format)
    """
    # Extract message logs for environment
    to_env = [
        get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
        for i in range(len(batch["message_log"]))
    ]
    task_names = [batch["task_name"][i] for i in range(len(batch["task_name"]))]

    # Group messages by task type
    task_groups = {}
    for i, task_name in enumerate(task_names):
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append((i, to_env[i]))

    # Calculate rewards for each task group concurrently
    futures = []
    future_to_indices = {}  # Map future to its corresponding indices
    for task_name, group in task_groups.items():
        if task_name not in task_to_env:
            raise ValueError(f"No environment found for task type: {task_name}")

        # Extract indices and messages for this group
        indices = [idx for idx, _ in group]
        messages = [msg for _, msg in group]

        # Get corresponding environment info
        env_info = [batch["extra_env_info"][i] for i in indices]

        # Submit task to environment and store future
        future = task_to_env[task_name].step.remote(messages, env_info)
        futures.append(future)
        future_to_indices[future] = indices

    results = ray.get(futures)
    all_rewards = []
    for future, result in zip(futures, results):
        indices = future_to_indices[future]
        _, _, task_rewards, _ = result

        # Store results with their original indices
        for idx, reward in zip(indices, task_rewards):
            all_rewards.append((idx, reward))

    # Sort results by original index to maintain order
    all_rewards.sort(key=lambda x: x[0])
    rewards = torch.tensor([reward for _, reward in all_rewards])

    return rewards, to_env


# ===============================================================================
# Training & Validation
# ===============================================================================


def grpo_train(
    policy: PolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: Dict[str, EnvironmentInterface],
    val_task_to_env: Optional[Dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: Optional[GRPOSaveState],
    master_config: MasterConfig,
):
    """Run GRPO training algorithm."""
    timer = Timer()
    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (hf framework backend)
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running

    # common config/state itmes
    step = grpo_save_state["step"]
    consumed_samples = grpo_save_state["consumed_samples"]
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    refit_buffer_size = master_config["policy"]["refit_buffer_size"]

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, refit_buffer_size)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    # Run grpo training (single-turn)
    batch: BatchedDataDict[DatumSpec]
    for batch in dataloader:
        print(
            f"\n{'=' * 25} Step {step + 1}/{min(len(dataloader), master_config['grpo']['max_num_steps'])} {'=' * 25}"
        )

        with timer.time("total_step_time"):
            # Prepare batch
            print("▶ Preparing batch...")
            with timer.time("data_processing"):
                # Repeat batch items
                repeated_batch = batch.repeat_interleave(
                    master_config["grpo"]["num_generations_per_prompt"]
                )
                # Convert LLMMessageLogType to FlatMessagesType for generation
                batched_flat, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                input_ids = batched_flat["token_ids"]
                # Create generation-specific input structure
                generation_input_data = BatchedDataDict[GenerationDatumSpec](
                    {
                        "input_ids": input_ids,
                        "input_lengths": input_lengths,
                    }
                )

            # Generate responses - this updates the LLMMessageLogType in repeated_batch
            print(f"▶ Generating responses for batch of size {len(input_ids)}...")
            with timer.time("prepare_for_generation"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
            with timer.time("generation"):
                repeated_batch, _, gen_metrics = generate_responses(
                    policy_generation,
                    generation_input_data,
                    repeated_batch,
                    tokenizer,
                    input_lengths,
                )
                policy_generation.finish_generation()

            # Calculate rewards & advantages based on the updated LLMMessageLogType
            print("▶ Calculating rewards...")
            with timer.time("reward_calculation"):
                rewards, _ = calculate_rewards(repeated_batch, task_to_env)

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

            # Run validation if it's a validation step
            if val_period > 0 and (step + 1) % val_period == 0:
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
                val_metrics, validation_timings = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step + 1,
                    master_config=master_config,
                )
                policy_generation.finish_generation()
                logger.log_metrics(
                    validation_timings, step + 1, prefix="timing/validation"
                )
                logger.log_metrics(val_metrics, step + 1, prefix="validation")

            ## Checkpointing
            consumed_samples += master_config["grpo"]["num_prompts_per_step"]
            if (
                master_config["checkpointing"]["enabled"]
                and (step + 1) % master_config["checkpointing"]["save_period"] == 0
            ):  # +1 because step is 0-indexed
                policy.prepare_for_training()

                is_last_checkpoint = (
                    min(len(dataloader), master_config["grpo"]["max_num_steps"])
                    - (step + 1)
                    < master_config["checkpointing"]["save_period"]
                )

                grpo_save_state["step"] = step + 1
                grpo_save_state["val_reward"] = val_metrics["accuracy"]
                grpo_save_state["consumed_samples"] = consumed_samples
                with timer.time("checkpointing"):
                    print(f"Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        step + 1, grpo_save_state, master_config
                    )
                    policy.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                        optimizer_path=os.path.join(
                            checkpoint_path, "policy", "optimizer"
                        ),
                        tokenizer_path=os.path.join(
                            checkpoint_path, "policy", "tokenizer"
                        ),
                        save_hf=is_last_checkpoint,
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)
                policy.offload_after_refit()

        # Logging
        # Log training data
        log_data = {"content": flat_messages["content"]}
        log_data["rewards"] = rewards.tolist()
        log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
        log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
        log_data["input_lengths"] = input_lengths.tolist()
        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

        print("\n📊 Training Results:")
        metrics = {
            "loss": train_results["loss"].numpy(),
            "reward": rewards.numpy(),
        }
        metrics.update(train_results["all_mb_metrics"])
        for k, v in metrics.items():
            if k == "num_valid_samples":
                metrics[k] = np.sum(v).item()
            else:
                metrics[k] = np.mean(v).item()
        metrics.update(gen_metrics)

        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        print(f"  • Loss: {metrics['loss']:.4f}")
        print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
        print(
            f"  • Mean Generation Length: {gen_metrics['mean_generation_length']:.4f}"
        )

        print("\n⏱️  Timing:")
        # Display total time first, separately
        total_time = timing_metrics.get("total_step_time", 0)
        print(f"  • Total step time: {total_time:.2f}s")

        # Display all other timing metrics
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
        if step >= master_config["grpo"]["max_num_steps"]:
            break


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    val_task_to_env: Dict[str, EnvironmentInterface],
    step: int,
    master_config: MasterConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

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

            # Convert LLMMessageLogType to FlatMessagesType for generation
            batched_flat, input_lengths = batched_message_log_to_flat_message(
                val_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
            )
            # Extract input IDs
            input_ids = batched_flat["token_ids"]
            # Create generation-specific input structure
            generation_input_data = BatchedDataDict(
                {
                    "input_ids": input_ids,
                    "input_lengths": input_lengths,
                }
            )

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            val_batch, generated_ids, gen_metrics = generate_responses(
                policy_generation,
                generation_input_data,
                val_batch,
                tokenizer,
                input_lengths,
                include_logprobs=False,
            )

            # Calculate rewards based on the updated LLMMessageLogType
            with timer.time("reward_calculation"):
                rewards, to_env = calculate_rewards(val_batch, val_task_to_env)

            total_rewards.extend(rewards.tolist())
            total_lengths.extend([len(ids) for ids in generated_ids])

            # Collect message logs for later display
            all_message_logs.extend(to_env)

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
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
