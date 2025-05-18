import argparse
import json
import os
import sys

current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pprint

from data_processors import (
    code_processor,
    genrm_processor,
    llm_judge_scp_116k_processor,
    math_processor,
)
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.llm_judge_environment import LLMJudgeEnvironment
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def build_task_processors(prompt_file: str, system_prompt_file: str):
    """Return a mapping from task-name -> (TaskDataSpec, processor_fn)."""
    # TODO: ykarnati - we need different specs for each task
    base_spec = TaskDataSpec(
        task_name="llm_judge_scp116k",
        prompt_file=prompt_file,
        system_prompt_file=system_prompt_file,
    )

    return {
        "llm_judge_scp116k": (base_spec, llm_judge_scp_116k_processor),
        "code": (base_spec, code_processor),
        "genrm": (base_spec, genrm_processor),
        "math": (base_spec, math_processor),
    }


def load_datasets(tokenizer: AutoTokenizer, data_config: DataConfig, task_processors):
    print("\n▶ Loading datasets...")

    dataset_path = data_config["dataset_path"]
    assert os.path.exists(dataset_path), f"{dataset_path} must exist"

    # Read JSONL dataset
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    continue

    train_dataset = AllTaskProcessedDataset(
        data,
        tokenizer,
        task_processors["llm_judge_scp116k"][0],
        task_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    # For this mock setup we use the same dataset for validation
    val_dataset = train_dataset

    return train_dataset, val_dataset


def create_envs(env_configs, task_keys, judge_handle=None):
    """Create Ray environment actors.

    Args:
        env_configs: Block from YAML under `env`
        task_keys: Iterable of task names
        judge_handle: ActorHandle or None
    Returns:
        task_to_env, val_task_to_env maps
    """
    judge_env = LLMJudgeEnvironment.options(
        runtime_env={
            "py_executable": LLMJudgeEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs["llm_judge"], judge_llm_handle=judge_handle)

    task_to_env = {k: judge_env for k in task_keys}
    return task_to_env, task_to_env


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    args, overrides = parser.parse_known_args()

    return args, overrides


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_llama_nemotron_mock_1B.yaml"
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

    if "llm_judge_generation" in config:
        llm_judge_tokenizer = get_tokenizer(config["llm_judge_generation"]["tokenizer"])
        config["llm_judge_generation"] = configure_generation_config(
            config["llm_judge_generation"], llm_judge_tokenizer
        )

    # 1️⃣  Load datasets first
    task_processors = build_task_processors(
        prompt_file=config["data"]["prompt_file"],
        system_prompt_file=config["data"]["system_prompt_file"],
    )

    train_dataset, val_dataset = load_datasets(
        tokenizer, config["data"], task_processors
    )

    # 2️⃣  Setup GRPO (creates cluster, policy & judge vLLMs)
    (
        policy,
        policy_generation,
        llm_judge_generation,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    # 3️⃣  Retrieve judge vLLM actor handle (if any)
    judge_handle = None
    if llm_judge_generation is not None:
        llm_judge_generation.prepare_for_generation()
        judge_handle = llm_judge_generation.worker_group.workers[0]

    # 4️⃣  Now create environment actors, passing the handle
    task_to_env, val_task_to_env = create_envs(
        config["env"], task_processors.keys(), judge_handle
    )

    # 5️⃣  Run training
    grpo_train(
        policy,
        policy_generation,
        train_dataloader,
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
