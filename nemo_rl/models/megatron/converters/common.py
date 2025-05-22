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

from collections import defaultdict
import einops
import numpy as np
from megatron.core import parallel_state
from nemo.lightning.io.state import (
    TransformCTX,
    _ModelState,
    StateDictTransform,
    _match_keys,
)
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations.accelerate import init_empty_weights

import nemo_rl.models.megatron.converters.qwen2 as qwen2_converter
import nemo_rl.models.megatron.converters.llama as llama_converter
from nemo_rl.models.megatron.refit_utils import get_global_param_key_to_local_key_map

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


def split_fc1_tp(ctx: TransformCTX, linear_fc1: torch.Tensor):
    # gate proj and up proj are mixed right now, and we need to reshape them
    # [ gate_tp0 ]     [ gate_tp0 ]
    # [  up_tp0  ] --\ [ gate_tp1 ] --\ (split gate)
    # [ gate_tp1 ] --/ [  up_tp0  ] --/ (split  up)
    # [  up_tp1  ]     [  up_tp1  ]
    megatron_config = ctx.source.config
    tp = megatron_config.tensor_model_parallel_size
    linear_fc1 = einops.rearrange(linear_fc1, "(t c d) a1 ->  c (t d) a1", c=2, t=tp)
    mlp_gate_proj_weight = linear_fc1[0]
    mlp_up_proj_weight = linear_fc1[1]
    return mlp_gate_proj_weight, mlp_up_proj_weight


def split_qkv_gpu(ctx: TransformCTX, linear_qkv: torch.Tensor):
    """
    Split interleave-concatenated qkv to q, k, v

    Example: export layer linear_qkv to HF {q|k|v}_proj
    """
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    # hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
    # when converting base model (linear_qkv), hidden size = megatron_config.hidden_size
    # when converting lora (linear_qkv.adapter.linear_out), hidden size = lora_r
    hidden_size = linear_qkv.size(-1)
    q_slice = torch.cat(
        [
            torch.arange(
                (heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size)
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size)
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size)

    return q_proj, k_proj, v_proj


def split_qkv_bias_gpu(ctx: TransformCTX, qkv_bias: torch.Tensor):
    """
    Split interleave-concatenated qkv bias to separate q, k, v bias

    Example: export layer linear_qkv bias to HF {q|k|v}_proj bias
    """
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_bias = qkv_bias.reshape([qkv_total_dim, head_size])
    q_slice = torch.cat(
        [
            torch.arange(
                (heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_bias = qkv_bias[q_slice].reshape(-1)
    k_bias = qkv_bias[k_slice].reshape(-1)
    v_bias = qkv_bias[v_slice].reshape(-1)

    return q_bias, k_bias, v_bias


def update_transforms_for_nemorl(export_transforms):
    # In place update
    for transform in export_transforms:
        if transform.transform.__name__ == "split_fc1":
            # Need to modify this transform to take into account the TP size
            transform.transform = split_fc1_tp
        elif transform.transform.__name__ == "split_qkv":
            # This transform previously moved qkv weights to cpu
            transform.transform = split_qkv_gpu
        elif transform.transform.__name__ == "split_qkv_bias":
            # This transform previously moved qkv weights to cpu
            transform.transform = split_qkv_bias_gpu
    return export_transforms


class MegatronToHFConverter:
    def __init__(self, hf_model_name, megatron_model):
        # We only care about the state_dict keys and the config, so we
        # don't need to load the model weights
        config = AutoConfig.from_pretrained(hf_model_name)
        with init_empty_weights():
            self.target_model = AutoModelForCausalLM.from_config(config)
        # TODO(yifu): inheritence for this?
        if "qwen" in hf_model_name.lower():
            self.export_mapping = qwen2_converter.get_export_mapping()
            self.export_transforms = qwen2_converter.get_export_transforms()
        elif "llama" in hf_model_name.lower():
            self.export_mapping = llama_converter.get_export_mapping()
            self.export_transforms = llama_converter.get_export_transforms(config)
        else:
            raise ValueError(
                f"No converter mapping and transforms found for {hf_model_name}"
            )

        self.export_transforms = update_transforms_for_nemorl(self.export_transforms)

        # Get all local keys across PP ranks
        local_keys = list(megatron_model.state_dict().keys())
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_world_size = torch.distributed.get_world_size(pp_group)

        all_local_keys_list = [None] * pp_world_size
        torch.distributed.all_gather_object(
            all_local_keys_list, local_keys, group=pp_group
        )
        all_local_keys = list({k for l in all_local_keys_list for k in l})

        global_key_map = get_global_param_key_to_local_key_map(
            megatron_model, megatron_model.config, all_local_keys
        )
        global_keys = list(global_key_map.keys())

        # Set the value of the state_dict to the megatron key name so that
        # StateDictTransform will set the value of the target state dict to
        # the megatron key name
        dummy_source_state_dict = {k: k for k in global_keys}
        ctx = TransformCTX(
            source=_ModelState(dummy_source_state_dict),
            source_state=dummy_source_state_dict,
            target=self.target_model,
            target_state=self._get_empty_state_dict(),
        )
        for key, val in self.export_mapping.items():
            ctx = StateDictTransform(key, val)(ctx)

        for transform in self.export_transforms:
            if type(transform.target_key) == tuple:
                for t in transform.target_key:
                    ctx = StateDictTransform(transform.source_key, t)(ctx)
            elif type(transform.source_key) == tuple:
                # TODO(yifu): handle many to one case with transform that just sets value as a list?
                ...
            else:
                ctx = StateDictTransform(transform.source_key, transform.target_key)(
                    ctx
                )

        hf_keys_to_megatron_keys = ctx.target_state
        megatron_keys_to_hf_keys = defaultdict(set)
        for hf_key, megatron_key in hf_keys_to_megatron_keys.items():
            if isinstance(megatron_key, list):
                for k in megatron_key:
                    megatron_keys_to_hf_keys[k].add(hf_key)
            else:
                megatron_keys_to_hf_keys[megatron_key].add(hf_key)
        self.megatron_keys_to_hf_keys = dict(megatron_keys_to_hf_keys)

    def _get_empty_state_dict(self, source_keys=None):
        if source_keys is None:
            # If source_keys is None, then we use all the target model keys
            target_keys = self.target_model.state_dict().keys()
        else:
            # Otherwise, we only use the target keys corresponding to the source_keys
            target_keys = set()
            for k in source_keys:
                target_keys = target_keys.union(self.megatron_keys_to_hf_keys[k])

        state_dict = {k: None for k in target_keys}
        return state_dict

    def convert(self, state_dict, megatron_config):
        source = _ModelState(state_dict)
        source.config = megatron_config
        ctx = TransformCTX(
            source=source,
            source_state=state_dict,
            target=self.target_model,
            target_state=self._get_empty_state_dict(list(state_dict.keys())),
        )
        for key, val in self.export_mapping.items():
            source_matches = _match_keys(list(state_dict.keys()), key)
            if source_matches.size == 1 and source_matches == np.array(None):
                continue
            ctx = StateDictTransform(key, val)(ctx)
        for transform in self.export_transforms:
            if type(transform.source_key) == tuple:
                source_keys = transform.source_key
            else:
                source_keys = (transform.source_key,)
            source_matches = _match_keys(list(state_dict.keys()), source_keys)
            if source_matches.size == 1 and source_matches == np.array(None):
                continue
            ctx = transform(ctx)

        converted_state_dict = ctx.target_state
        return converted_state_dict
