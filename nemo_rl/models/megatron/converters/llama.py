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

import einops
import torch

_GROUP_TO_RANKS_CACHE = {}


def split_qkv_llama(gathered_mcore_qkv_layer, cfg):
    hidden_size = cfg.hidden_size
    head_num = cfg.num_attention_heads
    num_query_groups = (
        cfg.num_query_groups or head_num
    )  # different num_query_groups for 70B

    head_size = cfg.kv_channels or (
        hidden_size // head_num
    )  # equivalent to hf's head_dim
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_weights = gathered_mcore_qkv_layer
    qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

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
    ## Example of slices
    ## 7b: num_query_groups = head_num = 32,
    ## q_slice = [0, 3, 6, 9 , ... 90, 93]
    ## k_slice = [1, 4, 7, 10, ... 91, 94]
    ## v_slice = [2, 5, 8, 11, ... 92, 95]
    ## 70b (with GQA): num_query_groups = 8, head_num = 64
    ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
    ## k_slice = [8, 18, 28, ... 68, 78]
    ## v_slice = [9, 19, 29, ... 69, 79]

    q_name = "model.layers.{gl}.self_attn.q_proj.weight"
    k_name = "model.layers.{gl}.self_attn.k_proj.weight"
    v_name = "model.layers.{gl}.self_attn.v_proj.weight"
    q = qkv_weights[q_slice].reshape(-1, hidden_size)
    k = qkv_weights[k_slice].reshape(-1, hidden_size)
    v = qkv_weights[v_slice].reshape(-1, hidden_size)

    return {q_name: q, k_name: k, v_name: v}


def split_fc1_gate_down_llama(gathered_mcore_fc1, cfg):
    # gate proj and up proj are mixed right now, and we need to reshape them
    # [ gate_tp0 ]     [ gate_tp0 ]
    # [  up_tp0  ] --\ [ gate_tp1 ] --\ (split gate)
    # [ gate_tp1 ] --/ [  up_tp0  ] --/ (split  up)
    # [  up_tp1  ]     [  up_tp1  ]
    tp = cfg.tensor_model_parallel_size
    gathered_mcore_fc1 = einops.rearrange(
        gathered_mcore_fc1, "(t c d) a1 ->  c (t d) a1", c=2, t=tp
    )
    mlp_gate_proj_weight = gathered_mcore_fc1[0]
    mlp_up_proj_weight = gathered_mcore_fc1[1]
    mlp_gate_proj_base_name = "model.layers.{gl}.mlp.gate_proj.weight"
    mlp_up_proj_base_name = "model.layers.{gl}.mlp.up_proj.weight"
    return {
        mlp_up_proj_base_name: mlp_up_proj_weight,
        mlp_gate_proj_base_name: mlp_gate_proj_weight,
    }


mcore_te_to_hf_llama = {
    "embedding.word_embeddings.weight": {"tp": 0, "hf": "model.embed_tokens.weight"},
    "decoder.final_layernorm.weight": {"hf": "model.norm.weight"},
    "output_layer.weight": {"tp": 0, "hf": "lm_head.weight"},
    "decoder.layers.{l}.self_attention.linear_proj.weight": {
        "tp": 1,
        "hf": "model.layers.{gl}.self_attn.o_proj.weight",
    },
    "decoder.layers.{l}.self_attention.linear_qkv.weight": {
        "tp": 0,
        "hf_func": split_qkv_llama,
    },
    "decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight": {
        "hf": "model.layers.{gl}.input_layernorm.weight"
    },
    "decoder.layers.{l}.mlp.linear_fc1.weight": {
        "tp": 0,
        "hf_func": split_fc1_gate_down_llama,
    },
    "decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight": {
        "hf": "model.layers.{gl}.post_attention_layernorm.weight"
    },
    "decoder.layers.{l}.mlp.linear_fc2.weight": {
        "tp": 1,
        "hf": "model.layers.{gl}.mlp.down_proj.weight",
    },
}
