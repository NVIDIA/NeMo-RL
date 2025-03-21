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
from typing import Any, Dict, Tuple, TypedDict

from torch.utils.data import DataLoader
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
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.utils.logger import Logger, LoggerConfig
from nemo_reinforcer.utils.timer import Timer


class SFTConfig(TypedDict):
    max_num_steps: int


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig


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


def setup(
    master_config: MasterConfig,
) -> Tuple[
    HfPolicy,
    RayVirtualCluster,
    DataLoader,
    AutoTokenizer,
    NLLLoss,
    MasterConfig,
    Logger,
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

    ## TODO: unify this with grpo
    data_cls = data_config["dataset_name"]
    if data_cls == "open_assistant":
        data = hf_datasets.OasstDataset(output_dir="/tmp/open_assistant")
    elif data_cls == "squad":
        data = hf_datasets.SquadDataset()
    else:
        raise ValueError(f"Unknown dataset class: {data_cls}")

    base_dataset = data.formatted_ds["train"]
    sft_task_spec = data.task_spec

    tokenizer = AutoTokenizer.from_pretrained(policy_config["model_name"])

    dataset = AllTaskProcessedDataset(
        base_dataset,
        tokenizer,
        sft_task_spec,
        sft_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=False,
        collate_fn=rl_collate_fn,  ## TODO: change this for sft! or make it more general
    )

    cluster = RayVirtualCluster(
        name="sft_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )

    policy = HfPolicy(cluster=cluster, config=policy_config)
    loss_fn = NLLLoss()

    logger = Logger(logger_config)

    return (
        policy,
        cluster,
        dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        sft_task_spec,
    )


def sft_train(
    policy, dataloader, tokenizer, loss_fn, master_config, logger, sft_task_spec
):
    # Run basic sft training
    timer = Timer()

    policy.prepare_for_training()

    for step, batch in enumerate(dataloader):
        timer.start("sft_train_step")

        timer.start("data_processing")
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
        timer.stop("data_processing")

        ## train_data.to("cpu")
        train_results = policy.train(train_data, loss_fn)
        timer.stop("sft_train_step")
        losses = train_results["loss"]
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        print(f"Step {step} completed. Loss: {losses[-1].item()}")

        logger.log_metrics(
            {"loss": losses[-1].item()},
            step,
            prefix="train",
        )
        logger.log_metrics(timing_metrics, step, prefix="timing/train")
        timer.reset()

        if step >= master_config["sft"]["max_num_steps"] - 1:
            break
