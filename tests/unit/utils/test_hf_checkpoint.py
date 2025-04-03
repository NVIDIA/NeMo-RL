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
import copy
import os
import pytest
import torch

from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from transformers import AutoTokenizer, AutoModelForCausalLM
from nemo_reinforcer.utils.hf_checkpoint import (
    load_checkpoint,
    save_checkpoint,
    ModelState,
    OptimizerState,
)

# Define basic test config
simple_policy_config = {
    "model_name": "meta-llama/Llama-3.2-1B",  # "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    "tokenizer_name": "meta-llama/Llama-3.2-1B",  # "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    "train_global_batch_size": 32,
    "train_micro_batch_size": 1,
    "logprob_batch_size": 1,
    "max_total_sequence_length": 1024,
    "precision": "float32",
}


@pytest.fixture
def test_experiment():
    model = torch.nn.ModuleList(
        [
            torch.nn.Linear(4, 4),
            torch.nn.LayerNorm(4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        ]
    ).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    return model, optimizer, scheduler


## TODO: check scope
@pytest.fixture(scope="module")
def cluster():
    """Create a virtual cluster for testing."""
    # Create a cluster with 1 GPU
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],  # [2],  # 1 node with 2 GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=1,  # 2,
        num_gpus_per_node=1,  # 2,  # Use available GPUs
        name="test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    tokenizer_name = simple_policy_config["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="function")
def policy(cluster, tokenizer):
    """Initialize the policy."""
    return HfPolicy(
        cluster=cluster,
        config=simple_policy_config,
        init_optimizer=False,
        init_reference_model=False,
    )


## recursively get the dummy state dict
## by replacing tensors with random ones of the same shape
def get_dummy_state_dict(state_dict, dummy_dict={}):
    for k in state_dict.keys():
        if isinstance(state_dict[k], dict):
            dummy_dict[k] = get_dummy_state_dict(state_dict[k], {})
        elif isinstance(state_dict[k], torch.Tensor):
            dummy_dict[k] = torch.randn(state_dict[k].shape)
        else:
            dummy_dict[k] = state_dict[k]
    return dummy_dict


## recursively check equality of two dictionaries
def check_dict_equality(dict1, dict2):
    for k in dict1.keys():
        if isinstance(dict1[k], dict):
            check_dict_equality(dict1[k], dict2[k])
        elif isinstance(dict1[k], torch.Tensor):
            assert torch.allclose(dict1[k], dict2[k])
        else:
            assert dict1[k] == dict2[k]


def test_model_state(test_experiment):
    test_model, _, _ = test_experiment
    model_state = ModelState(test_model)
    state_dict = model_state.state_dict()
    assert set(state_dict) == {"model"}

    ## relu has no parameters
    expected_keys = {
        "0.bias",
        "0.weight",
        "1.bias",
        "1.weight",
        "3.bias",
        "3.weight",
    }
    assert set(state_dict["model"].keys()) == expected_keys

    dummy_model_state_dict = get_dummy_state_dict(state_dict, {})

    ## update the model's state dict and verify that the model's parameters are updated
    model_state.load_state_dict(dummy_model_state_dict)
    new_model_state_dict = model_state.state_dict()
    check_dict_equality(new_model_state_dict, dummy_model_state_dict)


def test_optimizer_state(test_experiment):
    test_model, optimizer, scheduler = test_experiment

    optim_state = OptimizerState(test_model, optimizer, scheduler)
    state_dict = optim_state.state_dict()

    assert set(state_dict.keys()) == {"optim", "sched"}

    ## relu has no parameters
    expected_keys = {
        "0.bias",
        "0.weight",
        "1.bias",
        "1.weight",
        "3.bias",
        "3.weight",
    }

    assert set(state_dict["optim"]["state"].keys()) == expected_keys

    dummy_state_dict = get_dummy_state_dict(state_dict, {})

    optim_state.load_state_dict(dummy_state_dict)
    new_state_dict = optim_state.state_dict()
    check_dict_equality(new_state_dict, dummy_state_dict)


def test_save_and_load_model_only(test_experiment):
    test_model, _, _ = test_experiment
    save_checkpoint(test_model, "/tmp/test_model_only")
    assert os.path.exists("/tmp/test_model_only")
    assert not os.path.exists("/tmp/test_model_only-hf")
    assert set(os.listdir("/tmp/test_model_only")) == {".metadata", "__0_0.distcp"}


def test_save_and_load_model_and_optimizer(test_experiment):
    test_model, optimizer, scheduler = test_experiment
    for _ in range(5):
        scheduler.step()

    save_checkpoint(
        test_model,
        "/tmp/model_and_optimizer/model",
        optimizer,
        scheduler,
        optimizer_path="/tmp/model_and_optimizer/optimizer",
    )

    assert set(os.listdir("/tmp/model_and_optimizer/model")) == {
        ".metadata",
        "__0_0.distcp",
    }
    assert set(os.listdir("/tmp/model_and_optimizer/optimizer")) == {
        ".metadata",
        "__0_0.distcp",
    }

    ## modify the model, optimizer, and scheduler and verify that loading the checkpoint overrides the values
    new_linear = torch.nn.Linear(4, 4)
    new_linear.weight = torch.nn.Parameter(torch.ones([4, 4]).to("cuda"))
    new_linear.bias = torch.nn.Parameter(torch.ones(4).to("cuda"))
    new_model = torch.nn.ModuleList(
        [
            new_linear,
            torch.nn.LayerNorm(4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        ]
    ).to("cuda")

    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    new_scheduler = torch.optim.lr_scheduler.StepLR(
        new_optimizer, step_size=4, gamma=0.2
    )
    load_checkpoint(
        new_model,
        "/tmp/model_and_optimizer/model",
        new_optimizer,
        new_scheduler,
        optimizer_path="/tmp/model_and_optimizer/optimizer",
    )

    assert scheduler.state_dict() == new_scheduler.state_dict()
    check_dict_equality(new_model.state_dict(), test_model.state_dict())
    check_dict_equality(new_optimizer.state_dict(), optimizer.state_dict())


def test_save_and_load_hf_checkpoint(policy):
    ## warm up with a forward pass
    ## this is needed before saving a checkpoint because FSDP does some lazy initialization
    input_ids = torch.randint(0, 16000, (4, 128))  # 4 sequences, each of length 128
    attention_mask = torch.ones(4, 128)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)
    dummy_fwd_dict = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask,
            "labels": torch.randint(0, 16000, (4, 128)),
        }
    )
    policy.get_logprobs(dummy_fwd_dict)

    policy.save_checkpoint(
        "/tmp/test_hf_and_dcp",
        save_hf=True,
        save_torch_dist=True,
    )

    ## make sure we save both HF and DCP checkpoints
    assert set(os.listdir("/tmp/test_hf_and_dcp")) == {"__0_0.distcp", ".metadata"}
    ## 1B model has two shards
    assert set(os.listdir("/tmp/test_hf_and_dcp-hf")) == {
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    }

    coverted_model = AutoModelForCausalLM.from_pretrained("/tmp/test_hf_and_dcp-hf")
    original_model = AutoModelForCausalLM.from_pretrained(
        simple_policy_config["model_name"]
    )

    ## make sure this model matches the original
    check_dict_equality(coverted_model.state_dict(), original_model.state_dict())

    policy.worker_group.shutdown()
