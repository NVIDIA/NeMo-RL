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

import pytest
import torch

from datasets import load_dataset
from transformers import AutoTokenizer

from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.models.generation.vllm import VllmGeneration, VllmConfig
from nemo_reinforcer.tools.tools import BM25Retriever


# Define basic vLLM test config
basic_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": "meta-llama/Llama-3.2-1B",  # Small model for testing
    "dtype": "bfloat16",
    "max_new_tokens": 100,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "tool_map": {},
    "execute_code": True,
    "vllm_cfg": {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.3,
        "max_model_len": 1024,
    },
}


def configure_vllm_with_tokenizer(vllm_config, tokenizer, is_eval=False):
    """Apply tokenizer-specific configurations to vLLM config."""
    if is_eval:
        vllm_config["vllm_cfg"]["skip_tokenizer_init"] = False
        vllm_config["vllm_cfg"]["load_format"] = "auto"
    else:
        vllm_config["vllm_cfg"]["skip_tokenizer_init"] = True
        vllm_config["vllm_cfg"]["load_format"] = "dummy"
    vllm_config["pad_token"] = tokenizer.pad_token_id
    vllm_config["stop_token_ids"] = [tokenizer.eos_token_id]
    return vllm_config


@pytest.fixture(scope="module")
def cluster():
    """Create a virtual cluster for testing."""
    # Create a cluster with 1 node that has 1 GPU bundles
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],  # 1 node with 2 GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,  # Use available GPUs
        name="vllm-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    model_name = basic_vllm_test_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_vllm_execute_code(cluster, tokenizer):
    """Test that vLLM can call the code executor."""
    # Prepare test data
    codes = [
        ("<code>x = 3; y = 4</code>\nThis is some regular text.\n<code>x + y</code>\n"),
        ("<code>\ndef f(x):\n    return x * x\n\nf(2)\n</code>\n"),
    ]
    results = ["<result>7</result>", "\n<result>\n4\n</result>"]
    results = [code + result for code, result in zip(codes, results)]

    test_prompts = [code * 4 for code in codes]
    test_prompts = BatchedDataDict({"prompts": test_prompts})

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_vllm_with_tokenizer(vllm_config, tokenizer, is_eval=True)

    # Create vLLM generation
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Generate and check result
    output = vllm_generation.generate_text(test_prompts, greedy=True)

    assert output["texts"] == results, f"Got wrong output {output['texts']}"

    # Clean up
    vllm_generation.shutdown()


@pytest.mark.timeout(150)
def test_vllm_use_tool(cluster, tokenizer):
    """Test that vLLM can call the code executor."""
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
    test_prompts = BatchedDataDict({"prompts": test_prompts})

    # Retriever
    dataset = load_dataset("rahular/simple-wikipedia")
    documents = [sample["text"] for sample in dataset["train"]]
    retriever = BM25Retriever(documents, num_result=1)

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config["tool_map"] = {"retrieve": retriever}
    vllm_config = configure_vllm_with_tokenizer(vllm_config, tokenizer, is_eval=True)

    # Create vLLM generation
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Generate and check result
    output = vllm_generation.generate_text(test_prompts, greedy=True)

    assert output["texts"] == results, (
        f"Got wrong output {output['texts']}, expect {results}"
    )

    # Clean up
    vllm_generation.shutdown()
