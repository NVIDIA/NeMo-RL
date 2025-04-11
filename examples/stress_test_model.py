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

import argparse
import os
import random
import sys
import json
from pathlib import Path
from typing import TypedDict, cast
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import datetime

import torch
from torch.utils.data import DataLoader
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from transformers import AutoTokenizer
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.algorithms.utils import get_tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_reinforcer.distributed.virtual_cluster import (
    init_ray,
    RayVirtualCluster,
    ClusterConfig,
)
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.generation.interfaces import GenerationConfig
from nemo_reinforcer.models.generation.vllm import VllmGeneration
from nemo_reinforcer.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_reinforcer.algorithms.grpo import refit_policy_generation
from nemo_reinforcer.models.generation.interfaces import configure_generation_config


class DataConfig(TypedDict):
    numbers: dict[
        str, list[int]
    ]  # Dictionary with 'lengths' key containing list of integers
    random: dict[
        str, list[int]
    ]  # Dictionary with 'lengths' key containing list of integers
    literal: list[str]  # Lists of prompt strings


class Datum(TypedDict):
    prompt: str
    prompt_tokens: list[int]
    response_tokens: list[int]
    train_log_probs: list[float]
    inference_log_probs: list[float]
    error: float


class MasterConfig(TypedDict):
    generation: GenerationConfig
    data: DataConfig
    cluster: ClusterConfig
    policy: PolicyConfig
    checkpointing: CheckpointingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--test_refit",
        default=False,
        action="store_true",
        help="Whether to test refitting; (default: False which tests training and generation independently)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stress_test_results",
        help="Directory for output files (JSON and logs)",
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


def abbreviate_list(array, anchor: int = None) -> str:
    """Abbreviate a list for display purposes.

    If list length <= 10, returns the full list as a string.
    If list length > 10 and anchor provided, returns a truncated representation showing:
    - First three elements
    - Two elements before and after anchor
    - Last three elements
    - Ellipses (...) between non-adjacent sections

    Args:
        array: List or array-like object to abbreviate
        anchor: Index position to center the display around (optional)

    Returns:
        String representation of the abbreviated list
    """
    if len(array) <= 10:
        return str(array)

    if anchor is None or anchor < 0 or anchor >= len(array):
        # No valid anchor, just show first and last three elements
        return f"[{', '.join(str(x) for x in array[:3])}, ..., {', '.join(str(x) for x in array[-3:])}]"

    # Always show first and last three
    first_three = array[:3]
    last_three = array[-3:]

    # Calculate anchor window (anchor and two elements on each side)
    anchor_start = max(3, anchor - 2)  # Don't overlap with first three
    anchor_end = min(len(array) - 3, anchor + 2 + 1)  # Don't overlap with last three
    anchor_window = array[anchor_start:anchor_end]

    # Build the result
    result = [str(x) for x in first_three]

    # Add ellipsis between first three and anchor window if needed
    if anchor_start > 3:
        result.append("...")

    # Add anchor window
    result.extend([str(x) for x in anchor_window])

    # Add ellipsis between anchor window and last three if needed
    if anchor_end < len(array) - 3:
        result.append("...")

    # Add last three if they don't overlap with anchor window
    if anchor_end <= len(array) - 3:
        result.extend([str(x) for x in last_three])

    return f"[{', '.join(result)}]"


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
) -> tuple[
    VllmGeneration | HfPolicy,
    HfPolicy,
]:
    """Set up components for model evaluation.

    Initializes the VLLM model and data loader.

    Args:
        master_config: Configuration settings.
        dataset: Dataset to evaluate on.

    Returns:
        VLLM model, data loader, and config.
    """
    # Extract individual configs for easier access
    generation_config = master_config["generation"]
    cluster_config = master_config["cluster"]
    policy_config = master_config["policy"]

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="eval_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=2,  # 2 b/c we colocate policy and generation
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #           Model
    # ==========================
    print("\n▶ Setting up model...")
    # check backend
    backend = generation_config["backend"]
    assert backend in ("vllm", "hf"), "Only 'vllm' or 'hf' backend is supported"

    if backend == "hf":
        policy_generation = None
        print(f"  ✓ Using HF backend for generation with {policy_config['model_name']}")
    elif backend == "vllm":
        policy_generation = VllmGeneration(cluster=cluster, config=generation_config)
        # https://github.com/NVIDIA/reinforcer/issues/52
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        policy_generation.finish_inference()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    last_checkpoint_path = None
    if "checkpointing" in master_config:
        checkpointer = CheckpointManager(master_config["checkpointing"])
        last_checkpoint_path = checkpointer.get_latest_checkpoint_path()

    # TODO: generalize to mcore: https://github.com/NVIDIA/reinforcer/issues/133
    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        weights_path=Path(last_checkpoint_path) / "policy.pt"
        if last_checkpoint_path
        else None,
        init_optimizer=False,
    )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy_generation or policy,
        policy,
    )


def prepare_datasets(data_config, tokenizer):
    """Prepare test datasets from configuration."""
    datasets = {}

    if "numbers" in data_config:
        datasets["numbers"] = []
        for length in data_config["numbers"]["prompt_lengths"]:
            prompt = " ".join(str(i) for i in range(length))
            datasets["numbers"].append(
                {
                    "prompt": prompt,
                    "input_ids": tokenizer.encode(
                        prompt, return_tensors="pt", add_special_tokens=False
                    )[0],
                }
            )

    if "random" in data_config:
        datasets["random"] = []
        for length in data_config["random"]["prompt_lengths"]:
            # Exclude [] to avoid rich formatting issues
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@\\^_`{|}~ "
            prompt = "".join(random.choices(chars, k=length))
            datasets["random"].append(
                {
                    "prompt": prompt,
                    "input_ids": tokenizer.encode(
                        prompt, return_tensors="pt", add_special_tokens=False
                    )[0],
                }
            )

    if "literal" in data_config:
        datasets["literal"] = []
        for prompt in data_config["literal"]:
            if isinstance(prompt, str):
                datasets["literal"].append(
                    {
                        "prompt": prompt,
                        "input_ids": tokenizer.encode(
                            prompt, return_tensors="pt", add_special_tokens=False
                        )[0],
                    }
                )
            elif isinstance(prompt, list):
                message = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                datasets["literal"].append(
                    {
                        "prompt": message,
                        "input_ids": tokenizer(message, return_tensors="pt")[
                            "input_ids"
                        ][0],
                    }
                )
    if "message_jsonl_path" in data_config:
        datasets["message_jsonl_path"] = []
        with open(data_config["message_jsonl_path"], "r") as f:
            for line in f:
                prompt = json.loads(line)
                assert type(prompt) == list, "prompt must be a list"
                message = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                datasets["message_jsonl_path"].append(
                    {
                        "prompt": message,
                        "input_ids": tokenizer(message, return_tensors="pt")[
                            "input_ids"
                        ][0],
                    }
                )
    return datasets


def run_test_batch(
    generation,
    policy,
    tokenizer,
    dataset_items,
    batch_size,
    generation_kwargs,
    test_refit=False,
):
    """Run a test batch with the given configuration."""
    # Create data loader
    test_dataloader = DataLoader(
        dataset_items,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda data_batch: BatchedDataDict(
            {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [item["input_ids"] for item in data_batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ),
                "input_lengths": torch.tensor(
                    [len(item["input_ids"]) for item in data_batch]
                ),
                "prompts": [item["prompt"] for item in data_batch],
            }
        ),
    )

    print(f"Created dataloader with {len(test_dataloader)} batches")
    results = []

    # Process each batch
    for batch in test_dataloader:
        # If testing refit, perform the refit operation before each generation
        if test_refit and generation.__class__.__name__ == "VllmGeneration":
            # Refit the policy generation model to test refitting
            refit_policy_generation(policy, generation)
        else:
            # Generate outputs (only HF policies need to prevent optimizer/buffers from being offloaded)
            generation.prepare_for_inference(offload_optimizer_and_buffers=False)

        # Generate outputs
        outputs = generation.generate(batch, **generation_kwargs)
        generation.finish_inference()
        # Rename logprobs to avoid confusion
        outputs["inference_logprobs"] = outputs.pop("logprobs")

        # Get logprobs from policy
        policy.prepare_for_inference()
        fprop_logprob_data = BatchedDataDict(
            {
                "input_ids": outputs["output_ids"],
                "input_lengths": outputs["unpadded_sequence_lengths"],
            }
        )
        fprop_logprobs = policy.get_logprobs(fprop_logprob_data)["logprobs"]

        # Zero out logprobs for prompt and padded tokens
        for i, length in enumerate(batch["input_lengths"]):
            fprop_logprobs[i, :length] = 0

        for i, valid_seq_len in enumerate(outputs["unpadded_sequence_lengths"]):
            fprop_logprobs[i, valid_seq_len:] = 0

        outputs["training_logprobs"] = fprop_logprobs
        policy.offload_before_refit()
        policy.offload_after_refit()

        # Detokenize outputs
        detokenized_outputs = []
        for i in range(len(outputs["output_ids"])):
            gen_length = outputs["generation_lengths"][i].item()
            generated_ids = outputs["output_ids"][i, :gen_length]
            detokenized_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            detokenized_outputs.append(detokenized_text)

        outputs["detokenized_responses"] = detokenized_outputs
        results.append(batch | outputs)

    return results


def print_results(console, all_outputs, config, tokenizer):
    """Display test results in a readable format."""
    console.print("\n")
    console.rule("[bold cyan]EVALUATION RESULTS[/bold cyan]", style="cyan")
    console.print(f"Pad token ID: {tokenizer.pad_token_id}", style="dim")

    # Display detailed results for each test configuration
    for (
        dataset_name,
        max_new_tokens,
        generation_method,
        batch_size,
        refit_label,
    ), outputs in all_outputs.items():
        # Create header panel
        header = f"[bold]Dataset:[/bold] {dataset_name} | [bold]Max New Tokens:[/bold] {max_new_tokens} | [bold]Generation:[/bold] {generation_method} | [bold]Batch Size:[/bold] {batch_size} | [bold]Refit:[/bold] {refit_label}"
        console.print(Panel(header, style="cyan", expand=False))

        # Calculate statistics
        total_samples = sum(len(batch["generation_lengths"]) for batch in outputs)
        total_generation_tokens = sum(
            sum(length.item() for length in batch["generation_lengths"])
            for batch in outputs
        )
        total_input_tokens = sum(
            sum(length.item() for length in batch["input_lengths"]) for batch in outputs
        )

        avg_generation_length = (
            total_generation_tokens / total_samples if total_samples > 0 else 0
        )
        avg_input_length = (
            total_input_tokens / total_samples if total_samples > 0 else 0
        )

        # Display statistics table
        stats_table = Table(
            show_header=True, header_style="bold magenta", box=box.ROUNDED
        )
        stats_table.add_column("Statistic", style="dim")
        stats_table.add_column("Value", justify="right")
        stats_table.add_row("Average Input Length", f"{avg_input_length:.2f} tokens")
        stats_table.add_row(
            "Average Generation Length", f"{avg_generation_length:.2f} tokens"
        )
        stats_table.add_row("Total Samples", str(total_samples))
        console.print(stats_table)

        # Display sample outputs if available
        if outputs and len(outputs) > 0:
            print_sample_outputs(console, outputs, tokenizer)

        console.rule(style="dim")


def print_sample_outputs(console, outputs, tokenizer):
    """Display sample outputs from test runs."""
    console.print("\n[bold]Sample Outputs:[/bold]")

    # Loop through batches
    for batch_idx, sample_batch in enumerate(outputs):
        console.print(f"\n[bold]Batch {batch_idx + 1}:[/bold]", style="blue")

        for i in range(len(sample_batch["prompts"])):
            sample_table = Table(show_header=False, box=box.SIMPLE)
            sample_table.add_column("Field", style="green")
            sample_table.add_column("Content")

            # Truncate long text for display
            prompt = sample_batch["prompts"][i]
            prompt_display = f"{prompt[:100]}..." if len(prompt) > 100 else prompt

            response = sample_batch["detokenized_responses"][i]
            response_display = (
                f"{response[:100]}..." if len(response) > 100 else response
            )

            # Use input_length as anchor for abbreviating lists
            anchor = sample_batch["input_lengths"][i].item()

            # Display key information
            sample_table.add_row("Prompt", prompt_display)
            sample_table.add_row("Response", response_display)
            sample_table.add_row(
                "Input IDs",
                abbreviate_list(sample_batch["input_ids"][i].tolist(), anchor),
            )
            sample_table.add_row(
                "Output IDs",
                abbreviate_list(sample_batch["output_ids"][i].tolist(), anchor - 1),
            )
            sample_table.add_row("Input Length", str(anchor))
            sample_table.add_row(
                "Generation Length", str(sample_batch["generation_lengths"][i].item())
            )

            # Format and display logprobs
            inference_logprobs = [
                f"{x:.3f}" for x in sample_batch["inference_logprobs"][i].tolist()
            ]
            training_logprobs = [
                f"{x:.3f}" for x in sample_batch["training_logprobs"][i].tolist()
            ]
            sample_table.add_row(
                "Inference Logprobs", abbreviate_list(inference_logprobs, anchor)
            )
            sample_table.add_row(
                "Training Logprobs", abbreviate_list(training_logprobs, anchor)
            )

            # Calculate and display error
            avg_prob_mult_error = torch.mean(
                torch.exp(
                    torch.abs(
                        sample_batch["inference_logprobs"][i]
                        - sample_batch["training_logprobs"][i]
                    )
                )
            )
            sample_table.add_row("Avg Prob Mult Error", str(avg_prob_mult_error.item()))

            console.print(f"\n[bold]Sample {i + 1}:[/bold]")
            console.print(sample_table)


def print_summary_table(console, all_outputs, config):
    """Create and display summary table with aggregated results."""
    console.print("\n[bold cyan]SUMMARY RESULTS TABLE[/bold cyan]")
    summary_table = Table(
        show_header=True, header_style="bold white", box=box.SIMPLE_HEAD
    )
    summary_table.add_column("ISL (avg)", justify="right")
    summary_table.add_column("OSL (avg)", justify="right")
    summary_table.add_column("Max New Tokens", justify="right")
    summary_table.add_column("Data Type", justify="left")
    summary_table.add_column("Generation Method", justify="left")
    summary_table.add_column("Batch Size", justify="right")
    summary_table.add_column("Refit", justify="left")
    # TODO: Generalize https://github.com/NVIDIA/reinforcer/issues/133
    summary_table.add_column(
        f"HF vs {config['generation']['backend']}(gen)", justify="right"
    )

    # Aggregate results for summary
    aggregates = {}

    for (
        dataset_name,
        max_new_tokens,
        generation_method,
        batch_size,
        refit_label,
    ), outputs in all_outputs.items():
        key = (dataset_name, max_new_tokens, generation_method, batch_size, refit_label)
        aggregates[key] = {
            "input_lengths_sum": 0,
            "output_lengths_sum": 0,
            "total_samples": 0,
            "total_errors": 0,
            "valid_error_samples": 0,
        }

        # Collect aggregated data
        for batch in outputs:
            batch_size = len(batch["prompts"])
            aggregates[key]["total_samples"] += batch_size
            aggregates[key]["input_lengths_sum"] += torch.sum(
                batch["input_lengths"].to(torch.float32)
            ).item()
            aggregates[key]["output_lengths_sum"] += torch.sum(
                batch["generation_lengths"].to(torch.float32)
            ).item()

            # Calculate errors for each sample
            for i in range(len(batch["prompts"])):
                # Only compare generated tokens (excluding prompt and padding)
                valid_indices = torch.ones_like(
                    batch["training_logprobs"][i], dtype=torch.bool
                )
                valid_indices[: batch["input_lengths"][i]] = (
                    False  # Zero out prompt tokens
                )
                valid_indices[batch["unpadded_sequence_lengths"][i] :] = (
                    False  # Zero out padding tokens
                )

                inference_logprobs = batch["inference_logprobs"][i][valid_indices]
                training_logprobs = batch["training_logprobs"][i][valid_indices]

                # Calculate error as average probability multiplication factor
                if len(inference_logprobs) > 0:
                    error = torch.mean(
                        torch.exp(torch.abs(inference_logprobs - training_logprobs))
                    ).item()
                    aggregates[key]["total_errors"] += error
                    aggregates[key]["valid_error_samples"] += 1

    # Add rows to summary table
    for (
        dataset_name,
        max_new_tokens,
        generation_method,
        batch_size,
        refit_label,
    ), agg in aggregates.items():
        # Calculate averages
        avg_input_length = (
            agg["input_lengths_sum"] / agg["total_samples"]
            if agg["total_samples"] > 0
            else 0
        )
        avg_output_length = (
            agg["output_lengths_sum"] / agg["total_samples"]
            if agg["total_samples"] > 0
            else 0
        )

        # Calculate average error
        if agg["valid_error_samples"] > 0:
            avg_error = agg["total_errors"] / agg["valid_error_samples"]
            error_display = f"{avg_error:.4f}"
        else:
            error_display = "N/A (no generated tokens)"

        # Add row to table
        summary_table.add_row(
            f"{avg_input_length:.2f}",
            f"{avg_output_length:.2f}",
            str(max_new_tokens),
            dataset_name,
            generation_method,
            str(batch_size),
            refit_label,
            error_display,
        )

    console.print(summary_table)
    return aggregates


def save_results(console, all_outputs, config, output_dir):
    """Save results to JSON and log files."""
    # Prepare results for JSON
    json_results = {"config": config, "datasets": {}, "summary": []}

    # Get aggregated results for summary
    aggregates = print_summary_table(console, all_outputs, config)

    # Process all results for JSON output
    for (
        dataset_name,
        max_new_tokens,
        generation_method,
        batch_size,
        refit_label,
    ), outputs in all_outputs.items():
        # Create dataset entry
        key = f"{dataset_name}_{max_new_tokens}_{generation_method}_{batch_size}_{refit_label}"
        json_results["datasets"][key] = {
            "dataset_name": dataset_name,
            "max_new_tokens": max_new_tokens,
            "generation_method": generation_method,
            "batch_size": batch_size,
            "refit": refit_label,
            "samples": [],
        }

        # Add sample data
        for batch in outputs:
            for i in range(len(batch["prompts"])):
                # Only compare on generated tokens (excluding prompt and padding)
                valid_indices = torch.ones_like(
                    batch["training_logprobs"][i], dtype=torch.bool
                )
                valid_indices[: batch["input_lengths"][i]] = False
                valid_indices[batch["unpadded_sequence_lengths"][i] :] = False

                # Calculate error
                error = 0.0
                inference_logprobs = batch["inference_logprobs"][i][valid_indices]
                training_logprobs = batch["training_logprobs"][i][valid_indices]
                if len(inference_logprobs) > 0:
                    error = torch.mean(
                        torch.exp(torch.abs(inference_logprobs - training_logprobs))
                    ).item()

                # Add sample to JSON
                sample = {
                    "prompt": batch["prompts"][i],
                    "response": batch["detokenized_responses"][i],
                    "input_length": batch["input_lengths"][i].item(),
                    "generation_length": batch["generation_lengths"][i].item(),
                    "error": error,
                    "training_logprobs": batch["training_logprobs"][i].tolist(),
                    "inference_logprobs": batch["inference_logprobs"][i].tolist(),
                    "prompt_tokens": batch["prompt_tokens"][i].tolist()
                    if "prompt_tokens" in batch
                    else [],
                    "response_tokens": batch["response_tokens"][i].tolist()
                    if "response_tokens" in batch
                    else [],
                }
                json_results["datasets"][key]["samples"].append(sample)

    # Add summary data
    for (
        dataset_name,
        max_new_tokens,
        generation_method,
        batch_size,
        refit_label,
    ), agg in aggregates.items():
        avg_input_length = (
            agg["input_lengths_sum"] / agg["total_samples"]
            if agg["total_samples"] > 0
            else 0
        )
        avg_output_length = (
            agg["output_lengths_sum"] / agg["total_samples"]
            if agg["total_samples"] > 0
            else 0
        )

        avg_error = None
        if agg["valid_error_samples"] > 0:
            avg_error = agg["total_errors"] / agg["valid_error_samples"]

        # Add to summary
        json_results["summary"].append(
            {
                "avg_input_seq_length": avg_input_length,
                "avg_output_seq_length": avg_output_length,
                "dataset_name": dataset_name,
                "max_new_tokens": max_new_tokens,
                "generation_method": generation_method,
                "batch_size": batch_size,
                "refit": refit_label,
                "error": float(avg_error) if avg_error is not None else None,
                "total_samples": agg["total_samples"],
            }
        )

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate filenames with timestamp
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    model_name_sanitized = config["generation"]["model_name"].replace("/", "--")
    # TODO: Generalize https://github.com/NVIDIA/reinforcer/issues/133
    train_vs_gen = f"hf_vs_{config['generation']['backend']}-gen"
    log_file_path = os.path.join(
        output_dir, f"log_{model_name_sanitized}_{train_vs_gen}_{timestamp}.txt"
    )
    json_file_path = os.path.join(
        output_dir, f"results_{model_name_sanitized}_{train_vs_gen}_{timestamp}.json"
    )

    # Save files
    with open(json_file_path, "w") as f:
        json.dump(json_results, f, indent=2)

    console.save_text(log_file_path)

    console.print(f"\n[green]Results saved to:[/green]")
    console.print(f"[green]  - JSON: {json_file_path}[/green]")
    console.print(f"[green]  - Logs: {log_file_path}[/green]")


def main():
    """Main entry point."""
    # Parse arguments
    console = Console(record=True)
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "stress.yaml")

    config = OmegaConf.load(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        override_conf = OmegaConf.from_cli()
        print(f"Overrides: {override_conf}")
        config = OmegaConf.merge(config, override_conf)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    console.print("Final config:")
    console.print_json(json.dumps(config))

    # Init ray
    init_ray()

    cast(GenerationConfig, config["generation"])
    cast(DataConfig, config["data"])

    print("\n▶ Setting up tokenizer...")
    tokenizer = get_tokenizer(config["generation"]["model_name"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=not args.test_refit
    )

    assert tokenizer.chat_template

    # Prepare test datasets
    print("\n▶ Setting up data...")
    datasets = prepare_datasets(config["data"], tokenizer)
    assert datasets, f"No datasets found in config: {config['data']}"

    # Set up generation/policy
    print("\n▶ Creating generation backend and policy...")
    generation, policy = setup(config, tokenizer)

    print("\n▶ Testing different batch sizes and configurations...")
    # Storage for test results
    all_outputs = {}

    # Define test configurations
    batch_sizes = [generation.worker_group.world_size]  # TODO: re-enable single batch
    max_new_tokens_options = [512, config["generation"]["vllm_cfg"]["max_model_len"]]
    generation_methods = [
        (
            "greedy",
            {
                VllmGeneration: {
                    "greedy": True,
                    "sampling_param_overrides": {
                        "top_p": 1.0,
                        "top_k": 1,
                        "temperature": 0.0,
                        "max_tokens": None,  # Will be set in the loop
                    },
                },
                HfPolicy: {
                    "greedy": True,
                    "sampling_param_overrides": {
                        "top_p": 1.0,
                        "top_k": None,
                        "temperature": 1.0,
                        "max_new_tokens": None,  # Will be set in the loop
                    },
                },
            },
        ),
        (
            "default",
            {
                VllmGeneration: {
                    "greedy": False,
                    "sampling_param_overrides": {
                        "top_p": 1.0,
                        "top_k": -1,
                        "temperature": 1.0,
                        "max_tokens": None,  # Will be set in the loop
                    },
                },
                HfPolicy: {
                    "greedy": False,
                    "sampling_param_overrides": {
                        "top_p": 1.0,
                        "top_k": None,
                        "temperature": 1.0,
                        "max_new_tokens": None,  # Will be set in the loop
                    },
                },
            },
        ),
        # Enable this once https://github.com/NVIDIA/reinforcer/issues/69 is solved
        # (
        #     "top_p=0.95,temperature=0.7",
        #     {
        #         "sampling_param_overrides": {
        #             "top_p": 0.95,
        #             "top_k": -1,
        #             "temperature": 0.7,
        #             "max_tokens": None,  # Will be set in the loop
        #         },
        #     },
        # )
    ]

    # Count configurations to give a sense of progress
    total_configurations = (
        len(batch_sizes)
        * len(max_new_tokens_options)
        * len(generation_methods)
        * len(datasets)
    )
    i_config = 0

    ##################################
    # Run tests for all combinations #
    ##################################

    refit_label = "with_refit" if args.test_refit else "no_refit"
    for batch_size in batch_sizes:
        for max_new_tokens in max_new_tokens_options:
            for method_name, all_generation_kwargs in generation_methods:
                generation_kwargs = all_generation_kwargs[generation.__class__]
                # Set overrides specific to each generation method
                if config["generation"]["backend"] == "hf":
                    generation_kwargs["generation_batch_size"] = batch_size
                    generation_kwargs["sampling_param_overrides"]["max_new_tokens"] = (
                        max_new_tokens
                    )
                elif config["generation"]["backend"] == "vllm":
                    generation_kwargs["sampling_param_overrides"]["max_tokens"] = (
                        max_new_tokens
                    )
                else:
                    raise NotImplementedError(
                        f"Generation backend {config['generation']['backend']} not supported"
                    )

                # Test each dataset
                for dataset_name, dataset_items in datasets.items():
                    i_config += 1
                    print(
                        f"\n(Progress: {i_config}/{total_configurations}) Testing {dataset_name=} {batch_size=} {max_new_tokens=} {method_name=} {refit_label=}"
                    )

                    results = run_test_batch(
                        generation,
                        policy,
                        tokenizer,
                        dataset_items,
                        batch_size,
                        generation_kwargs,
                        test_refit=args.test_refit,
                    )
                    all_outputs[
                        (
                            dataset_name,
                            max_new_tokens,
                            method_name,
                            batch_size,
                            refit_label,
                        )
                    ] = results

    # Process and display results
    print_results(console, all_outputs, config, tokenizer)

    # Save results to files
    save_results(console, all_outputs, config, args.output_dir)


if __name__ == "__main__":
    main()
