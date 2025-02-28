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
from typing import Any, Dict, List, Union, Tuple

import torch
from datasets import Dataset

from nemo_reinforcer.data.interfaces import (
    TaskDataSpec,
    TaskDataProcessFnCallable,
    DatumSpec,
)
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict


# TODO @sahilj handle too-long prompts and masking them out throughout the whole process and renormalizing on loss
class AllTaskProcessedDataset:
    """Dataset for processing single or multi-task data with task-specific tokenization and processing.

    Args:
        dataset: Input dataset containing raw data
        tokenizer: Tokenizer for text processing
        default_task_data_spec: Default task processing specifications.
            In the case of single-task, this is the spec used for processing all entries.
            In the case of multi-task, any values not specified in the task-specific specs will be taken from the default spec.
        task_data_processors: Either a single TaskDataProcessFnCallable for single-task,
            or a dict mapping task names to (TaskDataSpec, TaskDataProcessFnCallable) for multi-task
        max_seq_length: Maximum sequence length for tokenized outputs
    """

    def __init__(
        self,
        dataset: Union[Dataset, Any],
        tokenizer,
        default_task_data_spec: TaskDataSpec,
        task_data_processors: Union[
            Dict[str, Tuple[TaskDataSpec, TaskDataProcessFnCallable]],
            TaskDataProcessFnCallable,
        ],
        max_seq_length=None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.default_task_data_spec = default_task_data_spec
        self.task_data_processors = task_data_processors
        self.max_seq_length = max_seq_length

        if isinstance(task_data_processors, dict):
            # apply defaults to all task data specs
            for task_name, (
                task_data_spec,
                task_data_processor,
            ) in task_data_processors.items():
                task_data_spec.copy_defaults(self.default_task_data_spec)

    def __len__(self):
        return len(self.dataset)

    def encode_single(self, text: Union[str, List[str]]) -> Tuple[List[int], int]:
        """Takes either a single string or a list of strings that represent multiple turns for the same conversation.

        Returns a single (concatenated) list of tokenized ids and the length of the tokenized ids.
        """
        if isinstance(text, str):
            text_ids = self.tokenizer.text_to_ids(text)
            return text_ids, len(text_ids)
        elif isinstance(text, list):
            text_ids = [self.tokenizer.text_to_ids(t) for t in text]
            return torch.cat(text_ids), sum(len(t) for t in text_ids)
        else:
            raise ValueError(
                f"text must be a string or a list of strings, got {type(text)}"
            )

    def __getitem__(self, idx: int) -> DatumSpec:
        """Return a single prompt."""
        entry = self.dataset[idx]

        if isinstance(self.task_data_processors, dict):
            task_name = entry["task_name"]

            assert task_name in self.task_data_processors, (
                f"task processor not provided for {task_name}. Provided processors: {self.task_data_processors.keys()}"
            )
            task_data_spec, task_data_processor = self.task_data_processors[task_name]
        else:
            task_data_spec = self.default_task_data_spec
            task_data_processor = self.task_data_processors

        datum_spec = task_data_processor(
            entry, task_data_spec, self.tokenizer, self.max_seq_length, idx
        )
        return datum_spec


def rl_collate_fn(data_batch: List[DatumSpec]) -> BatchedDataDict:
    """Collate function for RL training."""
    message_log = [datum_spec["message_log"] for datum_spec in data_batch]
    length = torch.tensor([datum_spec["length"] for datum_spec in data_batch])
    loss_multiplier = torch.tensor(
        [datum_spec["loss_multiplier"] for datum_spec in data_batch]
    )
    extra_env_info = [datum_spec["extra_env_info"] for datum_spec in data_batch]

    task_names = []
    for datum_spec in data_batch:
        task_names.append(datum_spec.get("task_name", None))

    idx = [datum_spec["idx"] for datum_spec in data_batch]
    batch_max_length = torch.ones_like(length) * length.max()

    output = BatchedDataDict(
        message_log=message_log,
        length=length,
        loss_multiplier=loss_multiplier,
        extra_env_info=extra_env_info,
        task_name=task_names,
        idx=idx,
        batch_max_length=batch_max_length,
    )
    return output
