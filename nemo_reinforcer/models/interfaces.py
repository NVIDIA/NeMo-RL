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
from abc import ABC, abstractmethod
from typing import Any, Dict

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.models.generation.interfaces import GenerationDatumSpec


class PolicyInterface(ABC):
    """Abstract base class defining the interface for RL policies."""

    @abstractmethod
    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get logprobs of actions from observations.

        Args:
            data: BatchedDataDict containing rollouts (tokens)

        Returns:
            BatchedDataDict containing:
                - logprobs: Tensor of logprobs of actions
        """
        pass

    @abstractmethod
    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get logprobs of actions from observations.

        Args:
            data: BatchedDataDict containing rollouts (tokens)

        Returns:
            BatchedDataDict containing:
                - logprobs: Tensor of logprobs of actions
        """
        pass

    @abstractmethod
    def train(self, data: BatchedDataDict, loss_fn: LossFunction) -> Dict[str, Any]:
        """Train the policy on a global batch of data.

        Args:
            data: BatchedDataDict containing rollouts (tokens)
        """
        pass

    @abstractmethod
    def prepare_for_training(self, *args, **kwargs):
        pass

    @abstractmethod
    def finish_training(self, *args, **kwargs):
        pass
