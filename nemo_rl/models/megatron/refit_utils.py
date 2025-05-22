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
import re
from typing import Dict, List, Tuple

import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
)
from nemo.collections.llm.gpt.model.base import GPTConfig

import nemo_rl.models.megatron.converters as model_converters


def get_tp_dim(model, param_name, named_modules_dict):
    # pass in named_modules_dict so we can get it ahead of time instead
    # of once for each param
    if not param_name.endswith(".weight") and not param_name.endswith(".bias"):
        return None

    prefix = ""
    if hasattr(model, "module"):
        prefix = "module."
        if hasattr(model.module, "module"):
            prefix = "module.module."
    key = prefix + ".".join(param_name.split(".")[:-1])
    module = named_modules_dict.get(key)
    if module is None:
        print(f"Module {key} not found in named_modules_dict")
        return None
    if hasattr(module, "parallel_mode"):
        # TE layers have parallel_mode we can check directly
        if module.parallel_mode == "column":
            return 0
        elif module.parallel_mode == "row":
            return 1
        else:
            return None
    elif isinstance(module, VocabParallelEmbedding) or isinstance(
        module, ColumnParallelLinear
    ):
        return 0
    elif isinstance(module, RowParallelLinear):
        return 1
    # TODO(yifu): moe layers
    else:
        return None


@torch.no_grad()
def get_global_param_key_to_local_key_map(
    model, model_cfg: GPTConfig, keys: List[Tuple[str, str]]
) -> Dict[str, Tuple[int, str]]:
    """Get a mapping from global parameter keys to local parameter keys.

    Args:
        model: The model to get the mapping for.
        model_cfg: The model configuration.
        keys: The local_keys to get the mapping for.

    Returns:
        A dictionary mapping global parameter keys to a tuple of (rank, local parameter key).
    """
    # Initialize pipeline parallel group information.
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_world_size = torch.distributed.get_world_size(pp_group)
    pp_global_rank_ids = model_converters.get_all_rank_ids_in_group(pp_group)

    # Build a mapping on each PP rank from a computed global key to the raw state dict key.
    # The global key is computed by replacing the local layer number (after "layers.")
    # with its corresponding global layer number (if applicable).
    local_map = {}
    for local_key in keys:
        if local_key not in model.state_dict():
            continue
        local_layer = model_converters.get_local_layer_num(local_key)
        if local_layer is not None:
            global_layer = model_converters.get_global_layer_num(local_key, model_cfg)
            # Replace the first occurrence of the digits after "layers." with the global layer number.
            global_key = re.sub(
                r"(?<=layers\.)\d+", str(global_layer), local_key, count=1
            )
        else:
            global_key = local_key
        local_map[global_key] = local_key

    # Gather the local maps from all PP ranks (only lightweight key info is gathered).
    all_maps = [None] * pp_world_size
    torch.distributed.all_gather_object(all_maps, local_map, group=pp_group)

    # Build the union over global keys and assign an owner (the rank with the smallest PP rank).
    union_global_map = {}
    for pp_rank, omap in enumerate(all_maps):
        for gk, raw_key in omap.items():
            if (
                gk not in union_global_map
                or pp_global_rank_ids[pp_rank] < union_global_map[gk][0]
            ):
                union_global_map[gk] = (pp_global_rank_ids[pp_rank], raw_key)
            else:
                print(
                    f"WARNING: {gk} already in union_global_map when gathering keys",
                    flush=True,
                )

    return union_global_map


@torch.no_grad()
def gather_params(
    model,
    param_name_to_rank_and_key,
):
    import time

    st = time.time()
    # Process each parameter (by its unique global key) one at a time.
    gathered_params = {}
    named_modules_dict = dict(model.named_modules())
    for gk in sorted(param_name_to_rank_and_key.keys()):
        owner_pp_global_rank, owner_raw_key = param_name_to_rank_and_key[gk]

        # Only the owner PP rank has the parameter locally.
        if torch.distributed.get_rank() == owner_pp_global_rank:
            param = model.state_dict()[owner_raw_key]

            # Check if param is TP-sharded
            tp_dim = get_tp_dim(model, owner_raw_key, named_modules_dict)

            # If the parameter is TP-sharded, gather its slices on GPU.
            if tp_dim is not None:
                tp_group = parallel_state.get_tensor_model_parallel_group()
                tp_world_size = torch.distributed.get_world_size(tp_group)
                gathered_slices = [
                    torch.empty_like(param) for _ in range(tp_world_size)
                ]
                torch.distributed.all_gather(gathered_slices, param, group=tp_group)
                full_param = torch.cat(gathered_slices, dim=tp_dim).to(torch.bfloat16)
            else:
                full_param = torch.clone(param).to(torch.bfloat16)

            # Use the original parameter key without conversion
            param_mapping = {gk: full_param}
        else:
            param_mapping = None  # Non-owner ranks will receive the tensors.

        # Broadcast the list of target parameter keys from the owner.
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if torch.distributed.get_rank() == owner_pp_global_rank:
            target_keys = [list(param_mapping.keys())]
        else:
            target_keys = [None]  # Placeholder to be filled by broadcast.

        torch.distributed.broadcast_object_list(
            target_keys, src=owner_pp_global_rank, group=pp_group
        )
        if "None" in target_keys[0]:
            continue

        # For each tensor, broadcast it individually.
        for target_key in target_keys[0]:
            if torch.distributed.get_rank() == owner_pp_global_rank:
                tensor_to_send = param_mapping[target_key]
            else:
                tensor_to_send = None
            # Broadcast tensor metadata (shape and dtype) to allocate GPU buffer on receiving ranks.
            meta = [None]
            if torch.distributed.get_rank() == owner_pp_global_rank:
                meta[0] = (tensor_to_send.shape, str(tensor_to_send.dtype))
            torch.distributed.broadcast_object_list(
                meta, src=owner_pp_global_rank, group=pp_group
            )
            shape, dtype_str = meta[0]
            dtype = getattr(torch, dtype_str.split(".")[-1])
            if torch.distributed.get_rank() != owner_pp_global_rank:
                tensor_to_send = torch.empty(
                    *shape, dtype=dtype, device=torch.cuda.current_device()
                )
            torch.distributed.broadcast(
                tensor_to_send, src=owner_pp_global_rank, group=pp_group
            )
            gathered_params[target_key] = tensor_to_send

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Time taken to gather params: {time.time() - st}")
    return gathered_params
