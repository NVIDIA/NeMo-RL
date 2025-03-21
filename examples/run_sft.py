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
import pprint
from typing import Dict, Any

from omegaconf import OmegaConf

from nemo_reinforcer.algorithms.sft import MasterConfig, sft_train, setup
from nemo_reinforcer.distributed.virtual_cluster import init_ray
from nemo_reinforcer.utils.config import load_config
from nemo_reinforcer.utils.logger import get_next_experiment_dir
from nemo_reinforcer.data import DataConfig, hf_datasets
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset
from nemo_reinforcer.data.interfaces import TaskDataSpec, DatumSpec
from nemo_reinforcer.data.llm_message_utils import get_formatted_message_log
from transformers import AutoTokenizer
from nemo_reinforcer.models.policy import PolicyConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SFT training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


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


def setup_data(data_config: DataConfig, policy_config: PolicyConfig):
    print("\n▶ Setting up data...")
    data_cls = data_config["dataset_name"]
    if data_cls == "open_assistant":
        data = hf_datasets.OasstDataset(output_dir="/tmp/open_assistant")
    elif data_cls == "squad":
        data = hf_datasets.SquadDataset()
    else:
        raise ValueError(f"Unknown dataset class: {data_cls}")
    print(
        f"  ✓ Training and validation datasets loaded with {len(data.formatted_ds['train'])} and {len(data.formatted_ds['validation'])} samples, respectively."
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

    val_dataset = AllTaskProcessedDataset(
        val_dataset,
        tokenizer,
        sft_task_spec,
        sft_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return train_dataset, val_dataset, tokenizer, sft_task_spec


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "sft.yaml")

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

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")

    init_ray()

    # setup data
    dataset, val_dataset, tokenizer, sft_task_spec = setup_data(
        config["data"], config["policy"]
    )
    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = setup(config, dataset, val_dataset)
    sft_train(
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
    )


if __name__ == "__main__":
    main()
