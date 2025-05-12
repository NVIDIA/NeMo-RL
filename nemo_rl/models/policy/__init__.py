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

from typing import Optional, TypedDict, Union

from nemo_rl.models.generation.interfaces import GenerationConfig


class DTensorConfig(TypedDict):
    enabled: bool
    cpu_offload: bool
    sequence_parallel: bool
    activation_checkpointing: bool
    tensor_parallel_size: int


class TokenizerConfig(TypedDict):
    name: str
    chat_template: str


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer: TokenizerConfig
    train_global_batch_size: int
    train_micro_batch_size: int
    learning_rate: float
    logprob_batch_size: int
    generation: Optional[GenerationConfig]
    precision: str
    dtensor_cfg: DTensorConfig
    make_sequence_length_divisible_by: int
    max_grad_norm: Optional[Union[float, int]]
    fsdp_offload_enabled: bool
    activation_checkpointing_enabled: bool
    refit_buffer_size_gb: int
