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


from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_math(data):
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data["answer"],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_deepscaler_dataset(seed=42):
    """Load and split the DeepScaler dataset into train and test sets."""
    # Load the original dataset
    original_ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")

    # Shuffle the dataset with the specified seed
    shuffled_ds = original_ds.shuffle(seed=seed)

    # Take 128 samples for test set
    test_ds = shuffled_ds.select(range(128))

    # Use the rest for training
    train_ds = shuffled_ds.select(range(128, len(shuffled_ds)))

    # Format the examples, removing original columns
    train_formatted = train_ds.map(format_math, remove_columns=train_ds.column_names)
    test_formatted = test_ds.map(format_math, remove_columns=test_ds.column_names)

    return {
        "train": train_formatted,
        "validation": test_formatted,
    }


class DeepScalerDataset:
    def __init__(self, seed: int = 42):
        """Initialize the DeepScaler dataset with train/test split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.formatted_ds = prepare_deepscaler_dataset(seed=seed)

        self.task_spec = TaskDataSpec(
            task_name="DeepScaler",
        )
