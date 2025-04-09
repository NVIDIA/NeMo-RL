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

from nemo_reinforcer.algorithms.dpo import MasterConfig, dpo_train, setup
from nemo_reinforcer.distributed.virtual_cluster import init_ray
from nemo_reinforcer.utils.config import load_config
from nemo_reinforcer.utils.logger import get_next_experiment_dir
from nemo_reinforcer.data import DataConfig, hf_datasets
from nemo_reinforcer.data.datasets import AllTaskProcessedDataset
from nemo_reinforcer.data.interfaces import TaskDataSpec, DatumSpec
from nemo_reinforcer.data.llm_message_utils import get_formatted_message_log
from transformers import AutoTokenizer
from nemo_reinforcer.models.policy import PolicyConfig

# from nemo_reinforcer.data.hf_datasets.helpsteer3 import HelpSteer3Dataset
from nemo_reinforcer.data.hf_datasets.dpo import DPODataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training with configuration")
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
def dpo_preprocessor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for DPO training."""
    if isinstance(datum_dict["prompt"], list):
        messages_chosen = datum_dict["prompt"]
        messages_rejected = datum_dict["prompt"]
    else:
        messages_chosen = [
            {
                "role": "user",
                "content": datum_dict["prompt"],
            },
        ]
        messages_rejected = [
            {
                "role": "user",
                "content": datum_dict["prompt"],
            },
        ]

    ## TODO: sometimes the context above includes assistant, but we don't want to train
    ## on that. Only want to train on the chosen and rejected responses... right? How do we ensure this?
    messages_chosen.append(
        {
            "role": "assistant",
            "content": datum_dict["chosen_response"],
        },
    )

    messages_rejected.append(
        {
            "role": "assistant",
            "content": datum_dict["rejected_response"],
        },
    )

    ## TODO: DO NOT APPLY CHAT TEMPLATE!
    message_log_chosen = get_formatted_message_log(
        messages_chosen, tokenizer, task_data_spec
    )
    message_log_rejected = get_formatted_message_log(
        messages_rejected, tokenizer, task_data_spec
    )

    length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
    length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    loss_multiplier = 1.0
    if max(length_chosen, length_rejected) > max_seq_length:
        # make smaller and mask out
        for message in message_log_chosen:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_chosen))
            ]
        for message in message_log_rejected:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_rejected))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log_chosen": message_log_chosen,
        "length_chosen": length_chosen,
        "message_log_rejected": message_log_rejected,
        "length_rejected": length_rejected,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def setup_data(data_config: DataConfig, policy_config: PolicyConfig):
    print("\n▶ Setting up data...")
    # data = HelpSteer3Dataset()
    # train_dataset = data.formatted_ds["train"]
    # val_dataset = data.formatted_ds["validation"]

    data = DPODataset(
        train_data_path=data_config["train_data_path"],
        val_data_path=data_config["val_data_path"],
    )
    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]

    dpo_task_spec = data.task_spec

    tokenizer = AutoTokenizer.from_pretrained(policy_config["model_name"])

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        dpo_task_spec,
        dpo_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        val_dataset,
        tokenizer,
        dpo_task_spec,
        dpo_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return train_dataset, val_dataset, tokenizer, dpo_task_spec


def main():
    """Main entry point."""
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "dpo.yaml")

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
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup data
    dataset, val_dataset, tokenizer, dpo_task_spec = setup_data(
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
        dpo_save_state,
        master_config,
    ) = setup(config, dataset, val_dataset)
    dpo_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        dpo_task_spec,
        checkpointer,
        dpo_save_state,
    )


if __name__ == "__main__":
    main()
