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

import torch
from megatron.core import parallel_state

_GROUP_TO_RANKS_CACHE = {}


def get_all_rank_ids_in_group(group):
    """Get all rank ids in a group."""
    if group in _GROUP_TO_RANKS_CACHE:
        return _GROUP_TO_RANKS_CACHE[group]

    curr_global_rank = int(torch.distributed.get_rank())
    group_size = torch.distributed.get_world_size(group=group)
    global_rank_tensor = torch.tensor(
        [curr_global_rank], dtype=torch.int, device=torch.cuda.current_device()
    )
    global_ranks = [
        torch.empty(1, dtype=torch.int, device=torch.cuda.current_device())
        for _ in range(group_size)
    ]
    torch.distributed.all_gather(global_ranks, global_rank_tensor, group=group)
    _GROUP_TO_RANKS_CACHE[group] = [
        int(global_ranks[i].item()) for i in range(group_size)
    ]
    return _GROUP_TO_RANKS_CACHE[group]


def get_local_layer_num(s):
    """Assumes layer number is preceeded by 'layers.'."""
    segments = s.split(".")
    number = None
    for i, segment in enumerate(segments):
        if segment == "layers":
            if segments[i + 1].isdigit():
                number = int(segments[i + 1])
                break
    return number


def get_global_layer_num(s, cfg):
    """Assumes layer number is preceeded by 'layers.'.

    Assumes pipeline model parallel size is set.
    In the state dict, the layer number is the local layer number (PP local).
    This function converts the local layer number to the global layer number.
    """
    local_layer_num = get_local_layer_num(s)
    global_layer_num = (
        parallel_state.get_pipeline_model_parallel_rank()
        * cfg.num_layers
        // parallel_state.get_pipeline_model_parallel_world_size()
        + local_layer_num
    )
    return global_layer_num


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"
