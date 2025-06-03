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

from nemo_rl.models.megatron.converters.llama import (
    split_fc1_gate_down_llama,
    split_qkv_llama,
)


def split_qkv_bias_qwen(gathered_mcore_qkv_layer, cfg):
    hidden_size = cfg.hidden_size
    head_num = cfg.num_attention_heads
    num_query_groups = (
        cfg.num_query_groups or head_num
    )  # different num_query_groups for GQA

    head_size = cfg.kv_channels or (
        hidden_size // head_num
    )  # equivalent to hf's head_dim
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_bias = gathered_mcore_qkv_layer
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

    q_name = "model.layers.{gl}.self_attn.q_proj.bias"
    k_name = "model.layers.{gl}.self_attn.k_proj.bias"
    v_name = "model.layers.{gl}.self_attn.v_proj.bias"
    q = qkv_bias[q_slice].reshape(-1)
    k = qkv_bias[k_slice].reshape(-1)
    v = qkv_bias[v_slice].reshape(-1)

    return {q_name: q, k_name: k, v_name: v}


mcore_te_to_hf_qwen2 = {
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
    "decoder.layers.{l}.self_attention.linear_qkv.bias": {
        "tp": 0,
        "hf_func": split_qkv_bias_qwen,
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
