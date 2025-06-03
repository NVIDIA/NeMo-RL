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

from typing import Any, Callable, Dict, Optional

from datasets import Dataset
from torch.utils.data import IterableDataset

from nemo_rl.data.packing.algorithms import PackingAlgorithm


class PackedDataset(IterableDataset):
    """Dataset wrapper that packs multiple samples into a single sample.

    This class is generic and can handle any type of dataset, regardless of the keys in the samples.
    It returns both the packed data and a list of the original samples that were packed together.
    """

    def __init__(
        self,
        dataset: Dataset,
        packer: PackingAlgorithm,
        prefetch_samples: int = 100,
        length_key: str = "length",
        length_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
    ):
        """Initialize the packed dataset.

        Args:
            dataset: Dataset to wrap
            packer: Packer implementation to use for packing samples
            prefetch_samples: Number of samples to prefetch from the parent dataset
            length_key: Key to use for determining sequence lengths (default: "length")
            length_fn: Optional function to compute length from a sample. If provided,
                       this overrides the length_key parameter.
        """
        self.dataset = dataset
        self.packer = packer
        self.prefetch_samples = prefetch_samples
        self.length_key = length_key
        self.length_fn = length_fn

    def _get_length(self, sample: Dict[str, Any]) -> int:
        """Get the length of a sample using either length_fn or length_key."""
        if self.length_fn is not None:
            return self.length_fn(sample)
        return sample[self.length_key]

    def __iter__(self):
        """Iterate over packed samples."""
        # Create an iterator for the dataset
        dataset_iter = iter(range(len(self.dataset)))

        while True:
            try:
                # Prefetch samples
                samples = []
                lengths = []

                for _ in range(self.prefetch_samples):
                    try:
                        idx = next(dataset_iter)
                        sample = self.dataset[idx]
                        samples.append(sample)
                        lengths.append(self._get_length(sample))
                    except StopIteration:
                        break

                if not samples:
                    break

                # Pack the samples
                packed_groups = self.packer.pack(lengths)

                # Yield each packed group
                for group in packed_groups:
                    if not group:
                        continue

                    yield [samples[i] for i in group]

            except StopIteration:
                break
