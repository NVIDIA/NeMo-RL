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
import math
import os
import pprint
from collections import defaultdict
from typing import Any, Dict

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.custom_datasets import python2cudac
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.cuda_environment import CudaEnvironment
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                           CUDA Problem Data Processor
# ===============================================================================


def cuda_data_processor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (from cuda_problems*.json) into a DatumSpec for the Cuda Environment."""
    problem = datum_dict["problem"]
    extra_env_info = {}  # Normally has ground truth

    message_log: LLMMessageLogType = []
    user_content = problem
    # Prepend prompt to user content if available
    if task_data_spec.prompt:
        user_content = task_data_spec.prompt.format(problem=problem)
    user_message = {
        "role": "user",
        "content": user_content,
    }
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=False,  # TODO: don't add generation prompt
        add_special_tokens=False,  # Usually False for user, True for system, but here we only have user
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message  # Store templated content
    message_log.append(user_message)

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
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "cuda"),  # Default task name
    }
    return output


def setup_data(
    tokenizer: AutoTokenizer, data_config: DataConfig, env_configs, grpo_config
):
    print("\n▶ Setting up data...")
    cuda_task_spec = TaskDataSpec(
        task_name="cuda",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )

    if data_config["dataset_name"] == "python2cudac":
        data_path = data_config.get(
            "dataset_path", "nemo_rl/data/custom_datasets/cuda_problems_tiny.json"
        )
        print(f"Loading custom CUDA dataset from: {data_path}")

        data = python2cudac.CudaProblemsDataset(
            json_file_path=data_path,
            seed=data_config.get("seed", 42),
            test_size=data_config.get("test_size", 0.05),
            duplicate_train_data=data_config.get(
                "duplicate_train_data", True
            ),  # Enable duplication so that we run through the train dataset multiple times until num_steps reached
            num_prompts_per_step=grpo_config.get("num_prompts_per_step"),
            max_num_steps=grpo_config.get("max_num_steps"),
        )
    else:
        raise ValueError(f"No processor for dataset {data_config['dataset_name']}.")

    task_data_processors = defaultdict(lambda: (cuda_task_spec, cuda_data_processor))
    task_data_processors["cuda"] = (cuda_task_spec, cuda_data_processor)

    cuda_env = CudaEnvironment.options(
        runtime_env={
            "py_executable": CudaEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs["cuda"])  # TODO: cuda key in env config
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        cuda_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        data.formatted_ds["validation"],
        tokenizer,
        cuda_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    task_to_env = defaultdict(lambda: cuda_env)
    task_to_env["cuda"] = cuda_env
    return dataset, val_dataset, task_to_env, task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "recipes",
            "cuda",
            "grpo-qwen2.5-coder-1.5b-instruct.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data - pass grpo config for potential duplication
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(
        tokenizer, config["data"], config["env"], config["grpo"]
    )  # Pass grpo config

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
