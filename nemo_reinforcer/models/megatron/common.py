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
from functools import partial
from typing import Iterable
import torch

from megatron.core.models.gpt import GPTModel

from nemo.tron.llm.gpt import get_batch
from nemo.tron.losses import masked_next_token_loss
from nemo.tron.state import GlobalState

from nemo_reinforcer.algorithms.loss_functions import LossFunction
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)


def forward_step_arbitrary_loss(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, loss_fn: LossFunction
):
    """Forward training step.

    Args:
        state (GlobalState): Global state for the run
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    # timers = state.timers
    straggler_timer = state.straggler_timer

    # timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        data_dict = next(data_iterator).to("cuda")
        input_ids = data_dict["input_ids"]
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            input_ids, 0, False, False, False
        )
        output_tensor = model(input_ids, position_ids, attention_mask)
        loss_data = data_dict
    # timers("batch-generator").stop()

    with straggler_timer:
        output_tensor = model(input_ids, position_ids, attention_mask)

    return output_tensor, partial(
        loss_fn,
        data=loss_data,
        vocab_parallel_rank=get_tensor_model_parallel_rank(),
        vocab_parallel_group=get_tensor_model_parallel_group(),
    )  # lambda x: (torch.sum(x), {'a': x}) #
