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

from copy import deepcopy

import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy.hf_policy import HfPolicy, PolicyConfig
from nemo_rl.tools.generation import generate_with_code_and_tools
from nemo_rl.tools.tools import BM25Retriever

MODEL_NAME = "meta-llama/Llama-3.2-1B"


# Define basic vLLM test config
basic_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": MODEL_NAME,
    "tokenizer_name": None,
    "dtype": "bfloat16",
    "max_new_tokens": 100,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.3,
        "max_model_len": 1024,
    },
}

basic_hf_test_config: PolicyConfig = {
    "model_name": MODEL_NAME,
    "tokenizer_name": None,
    "generation_batch_size": 1,
    "generation": {
        "backend": "hf",
        "max_new_tokens": 100,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
    },
    # Required training parameters
    "train_global_batch_size": 1,
    "train_micro_batch_size": 1,
    "learning_rate": 5e-6,
    "logprob_batch_size": 1,
    "max_new_tokens": 16,
    "do_sample": False,
    "precision": "float32",
    "activation_checkpointing_enabled": False,
    "fsdp_offload_enabled": False,
    "optimizer": {
        "name": "torch.optim.AdamW",
        "kwargs": {
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    },
    "dtensor_cfg": {"enabled": False},
}


@pytest.fixture(scope="module")
def cluster():
    """Create a virtual cluster for testing."""
    # Create a cluster with 1 node that has 1 GPU bundles
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],  # 1 node with 1 GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=1,  # Use available GPUs
        name="vllm-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def tokenizer():
    """Loads the tokenizer for the tests."""
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        f"Tokenizer loaded. Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id}), EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
    )
    return tokenizer


def test_vllm_execute_code(cluster, tokenizer):
    """Test that vLLM can call the code executor."""
    # Prepare test data
    codes = [
        "<code>x = 3; y = 4</code>\nThis is some regular text.\n<code>x + y</code>\n",
        "<code>\ndef f(x):\n    return x * x\n\nf(2)\n</code>\n",
    ]
    results = ["<result>7</result>", "\n<result>\n4\n</result>"]
    results = [code + result for code, result in zip(codes, results)]

    test_prompts = [code * 4 for code in codes]
    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
        padding_side="right",
    )
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
    batch = BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)

    # Create vLLM generation
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Generate and check result
    outputs = generate_with_code_and_tools(
        vllm_generation, batch, tokenizer, greedy=True
    )

    all_output_ids = outputs["output_ids"]
    logprobs = outputs["logprobs"]
    input_lengths = outputs["unpadded_sequence_lengths"] - outputs["generation_lengths"]
    output_lengths = outputs["unpadded_sequence_lengths"]
    input_ids = []
    output_ids = []
    for all_output_id, input_length, output_length in zip(
        all_output_ids, input_lengths, output_lengths
    ):
        input_ids.append(all_output_id[:input_length])
        output_ids.append(all_output_id[input_length:output_length])
    indices = torch.arange(all_output_ids.shape[-1])
    input_lengths = input_lengths.unsqueeze(-1)
    output_lengths = output_lengths.unsqueeze(-1)
    is_generated = (indices >= input_lengths) & (indices < output_lengths)

    input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    assert input_texts == test_prompts, "Unexpected modification to input texts"
    assert output_texts == results, f"Expect {results}, got wrong output {output_texts}"
    assert (logprobs[~is_generated] == 0.0).all(), (
        "Unexpected log probabilities on input tokens or paddings"
    )
    assert (logprobs[is_generated] != 0.0).all(), (
        "Generated tokens must have non-trivial log probabilities"
    )

    # Clean up
    vllm_generation.shutdown()


def test_hf_execute_code(cluster, tokenizer):
    """Test that Huggingface models can call the code executor."""
    # Prepare test data
    codes = [
        "<code>x = 3; y = 4</code>\nThis is some regular text.\n<code>x + y</code>\n",
        "<code>\ndef f(x):\n    return x * x\n\nf(2)\n</code>\n",
    ]
    results = ["<result>7</result>", "\n<result>\n4\n</result>"]
    results = [code + result for code, result in zip(codes, results)]

    test_prompts = [code * 4 for code in codes]
    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
        padding_side="right",
    )
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
    batch = BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )

    # Create separate configs for each policy
    hf_config = deepcopy(basic_hf_test_config)
    hf_config["generation"] = configure_generation_config(
        hf_config["generation"],
        tokenizer,  # is_eval=True
    )

    # Create vLLM generation
    hf_policy = HfPolicy(
        cluster, hf_config, tokenizer, init_reference_model=False, init_optimizer=False
    )

    # Generate and check result
    outputs = generate_with_code_and_tools(hf_policy, batch, tokenizer, greedy=True)

    all_output_ids = outputs["output_ids"]
    logprobs = outputs["logprobs"]
    input_lengths = outputs["unpadded_sequence_lengths"] - outputs["generation_lengths"]
    output_lengths = outputs["unpadded_sequence_lengths"]
    input_ids = []
    output_ids = []
    for all_output_id, input_length, output_length in zip(
        all_output_ids, input_lengths, output_lengths
    ):
        input_ids.append(all_output_id[:input_length])
        output_ids.append(all_output_id[input_length:output_length])
    indices = torch.arange(all_output_ids.shape[-1])
    input_lengths = input_lengths.unsqueeze(-1)
    output_lengths = output_lengths.unsqueeze(-1)
    is_generated = (indices >= input_lengths) & (indices < output_lengths)

    input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    assert input_texts == test_prompts, "Unexpected modification to input texts"
    assert output_texts == results, f"Expect {results}, got wrong output {output_texts}"
    assert (logprobs[~is_generated] == 0.0).all(), (
        "Unexpected log probabilities on input tokens or paddings"
    )
    assert (logprobs[is_generated] != 0.0).all(), (
        "Generated tokens must have non-trivial log probabilities"
    )

    # Clean up
    hf_policy.shutdown()


@pytest.mark.timeout(150)
def test_vllm_use_tool(cluster, tokenizer):
    """Test that vLLM can use tool in the code executor."""
    # Prepare test data
    codes = ["<code>retrieve('Jen-Hsun Huang')</code>\n"]
    results = [
        "\n<result>\n"
        "['Nvidia was established in 1993 by Jen-Hsun Huang, Curtis Priem, and Chris '\n"
        " 'Malachowsky. In 2000 Nvidia took intellectual possession of 3dfx, one of the '\n"
        " 'biggest GPU producers in 1990s.']\n"
        "</result>"
    ]
    results = [code + result for code, result in zip(codes, results)]

    test_prompts = [code * 4 for code in codes]
    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
        padding_side="right",
    )
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
    batch = BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )

    # Construct retriever
    dataset = load_dataset("rahular/simple-wikipedia")
    documents = [sample["text"] for sample in dataset["train"]]
    tool_map = {"retrieve": BM25Retriever(documents, num_result=1)}

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)

    # Create vLLM generation
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Generate and check result
    outputs = generate_with_code_and_tools(
        vllm_generation, batch, tokenizer, tool_map=tool_map, greedy=True
    )

    all_output_ids = outputs["output_ids"]
    input_lengths = outputs["unpadded_sequence_lengths"] - outputs["generation_lengths"]
    output_lengths = outputs["unpadded_sequence_lengths"]
    output_ids = []
    for all_output_id, input_length, output_length in zip(
        all_output_ids, input_lengths, output_lengths
    ):
        output_ids.append(all_output_id[input_length:output_length])

    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    assert output_texts == results, f"Expect {results}, got wrong output {output_texts}"

    # Clean up
    vllm_generation.shutdown()
