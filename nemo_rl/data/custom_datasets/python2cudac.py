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

import json
import math
from typing import Dict, Any, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets

from nemo_rl.data.interfaces import TaskDataSpec


def format_cuda_problem(data: Dict[str, Any]) -> Dict[str, Any]:
    """Basic formatter for cuda problem data. Assigns a default task name.

    TODO: potentially process other types of tasks as well.
    """
    output = {"problem": data["problem"]}
    output["task_name"] = data.get("task_name", "cuda")
    return output


def prepare_cuda_dataset(
    json_file_path: str,
    seed: int = 42,
    test_size: float = 0.05,
    duplicate_train_data: bool = False,
    num_prompts_per_step: Optional[int] = None,
    max_num_steps: Optional[int] = None,
) -> DatasetDict:
    """Load a custom JSON dataset, split it, and optionally duplicate the training set."""
    print(f"Loading dataset from {json_file_path}...")
    # Load the dataset from the JSON file, a list of dictionaries
    original_ds = Dataset.from_json(json_file_path)

    # Format the examples
    formatted_ds = original_ds.map(
        format_cuda_problem, remove_columns=original_ds.column_names
    )

    # Split into train and validation sets
    split_ds = formatted_ds.train_test_split(test_size=test_size, seed=seed)

    train_formatted = split_ds["train"]
    val_formatted = split_ds["test"]

    # --- Duplicate Training Data ---
    if (
        duplicate_train_data
        and num_prompts_per_step is not None
        and max_num_steps is not None
    ):
        original_train_size = len(train_formatted)
        total_samples_needed = num_prompts_per_step * max_num_steps
        if original_train_size > 0 and total_samples_needed > original_train_size:
            duplication_factor = math.ceil(total_samples_needed / original_train_size)
            print(
                f"Original train size: {original_train_size}. Samples needed: {total_samples_needed}. Duplicating training data {duplication_factor} times."
            )
            train_formatted = concatenate_datasets(
                [train_formatted] * duplication_factor
            )
            # Optional: Shuffle after duplication to mix the data
            train_formatted = train_formatted.shuffle(seed=seed)
            print(f"New duplicated train size: {len(train_formatted)}")
        else:
            print(
                f"Skipping duplication. Original train size ({original_train_size}) >= samples needed ({total_samples_needed}) or num_prompts/max_steps not provided."
            )
    # ----------------------------------------

    return DatasetDict(
        {
            "train": train_formatted,
            "validation": val_formatted,
        }
    )


class CudaProblemsDataset:
    def __init__(
        self,
        json_file_path: str,
        seed: int = 42,
        test_size: float = 0.05,
        duplicate_train_data: bool = False,
        num_prompts_per_step: Optional[int] = None,
        max_num_steps: Optional[int] = None,
    ):
        """Initialize the CudaProblemsDataset from a JSON file.

        Args:
            json_file_path: Path to the JSON file containing the dataset.
                            Expected format: List[Dict[str, Any]] where each dict
                            must contain at least a "problem" key.
            seed: Random seed for reproducible splitting and shuffling.
            test_size: Proportion of data to use for validation (0.0-1.0).
            duplicate_train_data: Whether to duplicate the training data to meet max_num_steps.
            num_prompts_per_step: Number of prompts per GRPO step (used for duplication calculation).
            max_num_steps: Maximum number of GRPO steps (used for duplication calculation).
        """
        self.formatted_ds = prepare_cuda_dataset(
            json_file_path=json_file_path,
            seed=seed,
            test_size=test_size,
            duplicate_train_data=duplicate_train_data,
            num_prompts_per_step=num_prompts_per_step,
            max_num_steps=max_num_steps,
        )

        self.task_spec = TaskDataSpec(
            task_name="cuda",  # Normally prompt/system_prompt is configued in main script
        )
