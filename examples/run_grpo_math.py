#!/usr/bin/env python3

import argparse
import os
import pprint

from omegaconf import OmegaConf
from typing import Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict

from nemo_reinforcer.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_reinforcer.distributed.virtual_cluster import init_ray
from nemo_reinforcer.utils.config import load_config
from nemo_reinforcer.utils.logger import get_next_experiment_dir
from nemo_reinforcer.data.interfaces import TaskDataSpec, DatumSpec, LLMMessageLogType
from nemo_reinforcer.data import DataConfig
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_reinforcer.environments.math_environment import MathEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


# ===============================================================================
#                             Math Data Processor
# ===============================================================================


# this processor expects the datum_dict to have a 'problem' key and an 'expected_answer' key
def math_data_processor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for the Math Environment."""
    problem = datum_dict["problem"]
    solution = str(datum_dict["expected_answer"])
    extra_env_info = {"ground_truth": solution}

    template = task_data_spec.custom_template
    message_log: LLMMessageLogType = []
    if task_data_spec.system_prompt:
        sys_message = {"role": "system", "content": task_data_spec.system_prompt}
        message = tokenizer.apply_chat_template(
            [sys_message],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][
            0
        ]
        message_log.append(sys_message)
    user_message = {
        "role": "user",
        "content": task_data_spec.prompt.format(problem),
    }
    message = tokenizer.apply_chat_template(
        [user_message],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
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
        "task_name": datum_dict["task_name"],
    }
    return output


def setup_data(data_config: DataConfig, policy_config: PolicyConfig, env_configs):
    print("\n▶ Setting up data...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    base_dataset = load_dataset("json", data_files=data_config["dataset_name"])["train"]
    tokenizer = AutoTokenizer.from_pretrained(policy_config["model_name"])

    task_data_processors = defaultdict(lambda: (math_task_spec, math_data_processor))
    task_data_processors["math"] = (math_task_spec, math_data_processor)

    math_env = MathEnvironment.options(
        runtime_env={"py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE}
    ).remote(env_configs["math"])
    dataset = AllTaskProcessedDataset(
        base_dataset,
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    if "val_dataset_name" in data_config and data_config["val_dataset_name"]:
        val_dataset = load_dataset("json", data_files=data_config["val_dataset_name"])[
            "train"
        ]
        val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            math_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    task_to_env = defaultdict(lambda: math_env)
    task_to_env["math"] = math_env
    return dataset, val_dataset, task_to_env, task_to_env, tokenizer


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = OmegaConf.merge(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")

    init_ray()

    # setup data
    dataset, val_dataset, task_to_env, val_task_to_env, tokenizer = setup_data(
        config["data"], config["policy"], config["env"]
    )
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
    ) = setup(config, dataset, val_dataset)
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
