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
import ray

from transformers import AutoTokenizer

from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.models.generation.vllm import VllmGeneration, VllmConfig


# Define basic vLLM test config
basic_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": "meta-llama/Llama-3.2-1B",  # Small model for testing
    "dtype": "bfloat16",
    "max_new_tokens": 10,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "vllm_cfg": {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.3,
        "max_model_len": 1024,
    },
}


def configure_vllm_with_tokenizer(vllm_config, tokenizer):
    """Apply tokenizer-specific configurations to vLLM config."""
    vllm_config["skip_tokenizer_init"] = True
    vllm_config["load_format"] = "auto"
    vllm_config["pad_token"] = tokenizer.pad_token_id
    vllm_config["stop_token_ids"] = [tokenizer.eos_token_id]
    return vllm_config


@pytest.fixture(scope="module")
def check_vllm_available():
    """Skip tests if vLLM is not installed."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vLLM not installed")


@pytest.fixture(scope="module")
def cluster():
    """Create a virtual cluster for testing."""
    # Create a cluster with 1 node that has 2 GPU bundles
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],  # 1 node with 2 GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,  # Use available GPUs
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


@pytest.fixture(scope="function")
def policy(cluster, tokenizer, check_vllm_available):
    """Initialize the vLLM policy."""
    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_vllm_with_tokenizer(vllm_config, tokenizer)
    policy = VllmGeneration(cluster, vllm_config)
    yield policy

    # Ensure policy is properly shutdown
    try:
        policy.shutdown()
        # Force garbage collection to help release resources
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during policy cleanup: {e}")


@pytest.fixture(scope="function")
def test_input_data(tokenizer):
    """Create test input data for inference."""
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    # Tokenize prompts
    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input lengths from attention mask
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)

    # Create input data dictionary
    return BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )


def test_vllm_missing_required_config_key(cluster, check_vllm_available):
    """Test that an assertion error is raised when a required config key is missing."""
    # Create a config missing a required key by removing 'model_name'
    incomplete_config = basic_vllm_test_config.copy()
    del incomplete_config["model_name"]  # Remove a required key

    # Also need to ensure skip_tokenizer_init and load_format are there
    # since these are checked in VllmConfig.__annotations__
    incomplete_config["skip_tokenizer_init"] = True
    incomplete_config["load_format"] = "auto"

    # Attempt to initialize VllmGeneration with incomplete config - should raise AssertionError
    with pytest.raises(AssertionError) as excinfo:
        VllmGeneration(cluster, incomplete_config)

    # Verify the error message contains information about the missing key
    error_message = str(excinfo.value)
    assert "Missing required keys in VllmConfig" in error_message
    assert "model_name" in error_message, (
        "Error should mention the missing 'model_name' key"
    )
    print(f"Successfully caught missing config key with error: {error_message}")


def test_vllm_policy_generation(policy, test_input_data, tokenizer):
    """Test vLLM policy generation capabilities."""
    # Test generation
    print("Testing generation...")
    outputs = policy.generate(test_input_data)

    # Validate outputs format
    assert "output_ids" in outputs, "output_ids not found in generation output"
    assert "logprobs" in outputs, "logprobs not found in generation output"
    assert "generation_lengths" in outputs, (
        "generation_lengths not found in generation output"
    )
    assert "unpadded_sequence_lengths" in outputs, (
        "unpadded_sequence_lengths not found in generation output"
    )

    # Validate outputs shape and content
    assert outputs["output_ids"].shape[0] == len(test_input_data["input_ids"]), (
        "Wrong batch size in output"
    )
    assert outputs["generation_lengths"].shape[0] == len(
        test_input_data["input_ids"]
    ), "Wrong batch size in generation_lengths"

    # Decode and check outputs
    generated_sequences = outputs["output_ids"]
    generated_texts = tokenizer.batch_decode(
        generated_sequences, skip_special_tokens=True
    )

    print(f"Generated texts: {generated_texts}")

    # All texts should have a non-zero length and be longer than inputs
    assert all(len(text) > 0 for text in generated_texts), (
        "Some generated texts are empty"
    )


@pytest.mark.timeout(140)
def test_vllm_generation_with_hf_training(cluster, tokenizer):
    """1. Use vLLM for generation
    2. Use HF policy for training and logprob computation

    This test validates that the two policies can work together.
    """
    from nemo_reinforcer.models.policy.hf_policy import HfPolicy
    from tests.unit.test_utils import nll_loss

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_vllm_with_tokenizer(vllm_config, tokenizer)

    # Create HF-specific config with required parameters
    hf_config = {
        "model_name": basic_vllm_test_config["model_name"],
        # Required training parameters
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "max_new_tokens": 16,
        "do_sample": False,
        "precision": "float32",
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 5e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
    }

    vllm_policy = None
    hf_policy = None

    try:
        prompts = [
            "Write a story about a magical forest",
            "Explain how photosynthesis works",
            "What are the benefits of exercise?",
            "Describe the water cycle",
            "What is the capital of France?",
            "Who is the president of the USA?",
            "What is the capital of the moon?",
            "Where is the sun?",
        ]

        # Tokenize the prompts the same way as in test_hf_ray_policy
        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
            padding_side="right",
        )
        # Calculate input lengths from attention mask
        input_lengths = tokenized["attention_mask"].sum(dim=1).to(torch.int32)

        test_input_data = BatchedDataDict(
            {
                "input_ids": tokenized["input_ids"],
                "input_lengths": input_lengths,
            }
        )

        # Create both policies
        print("Creating vLLM policy...")
        vllm_policy = VllmGeneration(cluster, vllm_config)
        vllm_policy.finish_generation()

        print("Creating HF policy...")
        hf_policy = HfPolicy(cluster, hf_config)

        print(f"refitting vllm policy...")
        ipc_handles = hf_policy.get_weights_ipc_handles()
        vllm_policy.prepare_for_generation()
        vllm_policy.update_weights(ipc_handles)

        # Step 1: Use vLLM for generation
        print("Using vLLM policy for fast generation...")
        generation_results = vllm_policy.generate(test_input_data)
        vllm_policy.finish_generation()
        # Validate generation outputs
        assert "output_ids" in generation_results, (
            "output_ids not found in vLLM generation output"
        )
        assert "logprobs" in generation_results, (
            "logprobs not found in vLLM generation output"
        )

        # Decode generations
        generated_texts = tokenizer.batch_decode(
            generation_results["output_ids"], skip_special_tokens=True
        )
        print(f"vLLM generated texts: {generated_texts}")

        # Run logprob calculation with HF policy to verify

        fprop_logprob_data = BatchedDataDict(
            {
                "input_ids": generation_results["output_ids"],
                "input_lengths": generation_results["unpadded_sequence_lengths"],
            }
        )
        # Get logprobs from HF policy
        hf_policy.prepare_for_lp_inference()
        fprop_results = hf_policy.get_logprobs(fprop_logprob_data)
        # Zero out logprobs for input tokens

        print(f"HF logprobs: {fprop_results['logprobs']}")
        print(f"vLLM logprobs: {generation_results['logprobs']}")

        # Validate that the logprobs are correct (comparing vLLM generation logprobs with HF computed logprobs)

        # Create a mask for padding tokens to only include tokens up to generation_lengths
        padding_mask = torch.zeros_like(
            generation_results["logprobs"], dtype=torch.bool
        )
        for i, (input_len, total_valid_len) in enumerate(
            zip(
                test_input_data.get("input_lengths"),
                generation_results["unpadded_sequence_lengths"],
            )
        ):
            padding_mask[i, input_len:total_valid_len] = True

        abs_diff = torch.abs(generation_results["logprobs"] - fprop_results["logprobs"])
        masked_abs_diff = abs_diff.masked_select(padding_mask)
        avg_prob_mult_error = (
            torch.mean(torch.exp(masked_abs_diff))
            if masked_abs_diff.numel() > 0
            else torch.tensor(0.0)
        )

        print(f"Average probability multiplicative error: {avg_prob_mult_error}")
        assert avg_prob_mult_error <= 1.043, "vLLM and HF logprobs should closely match"

        # Step 2: Prepare simplified training data (smaller and with padding removed to prevent OOM)
        # Use a very small sequence for training to ensure it works
        max_seq_len = min(40, generation_results["output_ids"].shape[1])
        # cap generation lengths to max_seq_len
        generation_results["unpadded_sequence_lengths"] = torch.clamp(
            generation_results["unpadded_sequence_lengths"], max=max_seq_len
        )

        train_input_ids = generation_results["output_ids"][:, :max_seq_len]
        token_loss_mask = torch.ones_like(train_input_ids)
        # Only compute loss on generated tokens, not input
        input_len = test_input_data.get("input_ids").size(1)
        token_loss_mask[:, :input_len] = 0

        for idx, length in enumerate(generation_results["unpadded_sequence_lengths"]):
            token_loss_mask[idx, length:] = 0

        train_data = BatchedDataDict(
            {
                "input_ids": train_input_ids,
                "input_lengths": generation_results["unpadded_sequence_lengths"],
                "token_loss_mask": token_loss_mask,
            }
        )

        # Step 3: Try a minimal training step with HF policy
        print("Training with HF policy (single step)...")
        hf_policy.prepare_for_training()

        # Just do one training step to verify it works
        results = hf_policy.train(train_data, nll_loss)
        print(f"Training loss: {results['loss']}")

        hf_policy.finish_training()
        hf_policy.offload_after_refit()

        # Step 4: Use vLLM for generation again to complete the workflow
        print("Using vLLM for generation again...")
        vllm_policy.prepare_for_generation()
        final_generation = vllm_policy.generate(test_input_data)
        assert "output_ids" in final_generation, (
            "Final generation should contain output_ids"
        )

        print("Successfully demonstrated vLLM generation + HF training workflow!")

    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if vllm_policy:
            vllm_policy.shutdown()
        if hf_policy and hasattr(hf_policy, "shutdown"):
            hf_policy.shutdown()


def test_vllm_policy_tensor_parallel(cluster, tokenizer):
    """Test vLLM policy with tensor parallelism > 1."""
    # Configure with tensor_parallel_size=2
    tp_config = basic_vllm_test_config.copy()
    tp_config = configure_vllm_with_tokenizer(tp_config, tokenizer)
    tp_config["tensor_parallel_size"] = 2

    # Ensure we specify the distributed executor backend
    tp_config["vllm_kwargs"] = {"distributed_executor_backend": "ray"}

    vllm_policy = None
    try:
        vllm_policy = VllmGeneration(cluster, tp_config)

        # Create simple test input
        test_prompts = ["Hello, my name is", "The capital of France is"]
        encodings = tokenizer(
            test_prompts,
            padding="max_length",
            max_length=10,
            truncation=True,
            return_tensors="pt",
            padding_side="right",
        )

        test_input_data = BatchedDataDict(
            {
                "input_ids": encodings["input_ids"],
                "input_lengths": encodings["attention_mask"].sum(dim=1).to(torch.int32),
            }
        )

        # Test generation with tensor parallelism
        outputs = vllm_policy.generate(test_input_data)

        vllm_policy.finish_generation()
        vllm_policy.prepare_for_generation()
        # Validate outputs
        # Test generation with tensor parallelism
        outputs = vllm_policy.generate(test_input_data)

        assert "output_ids" in outputs, "output_ids not found in generation output"
        assert outputs["output_ids"].shape[0] == 2, "Wrong batch size in output"

        # Decode and check output
        generated_text = tokenizer.decode(
            outputs["output_ids"][0], skip_special_tokens=True
        )
        print(f"Generated text with TP=2: {generated_text}")
        assert len(generated_text) > 0, "Generated text is empty"

    finally:
        # Clean up resources
        if vllm_policy:
            vllm_policy.shutdown()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_vllm_policy_weight_update(cluster, tokenizer, tensor_parallel_size):
    """Test that weights can be updated from HF to vLLM policy."""
    # Create HF policy
    from nemo_reinforcer.models.policy.hf_policy import HfPolicy

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_vllm_with_tokenizer(vllm_config, tokenizer)
    vllm_config["tensor_parallel_size"] = tensor_parallel_size

    # Add vllm_kwargs only if using tensor parallelism
    if tensor_parallel_size > 1:
        vllm_config["vllm_kwargs"] = {"distributed_executor_backend": "ray"}

    # Create HF-specific config with required parameters
    hf_config = {
        "model_name": basic_vllm_test_config["model_name"],
        # Required training parameters
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "max_new_tokens": 16,
        "do_sample": False,
        "precision": "float32",
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 5e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
    }

    hf_policy = HfPolicy(cluster, hf_config)
    print(f"hf_policy created: {hf_policy}", flush=True)
    # hf_policy.finish_training()
    vllm_policy = VllmGeneration(cluster, vllm_config)
    print(
        f"vllm_policy created with tensor_parallel_size={tensor_parallel_size}: {vllm_policy}",
        flush=True,
    )

    # Test generation with tensor parallelism
    vllm_policy.finish_generation()
    # hf_policy.prepare_for_training()

    # Zero out the weights in the HF model via workers
    ray.get(
        [worker.zero_out_weights.remote() for worker in hf_policy.worker_group.workers]
    )
    print("Zeroed out weights in HF policy")
    # Get device IDs
    training_device_id = ray.get(
        hf_policy.worker_group.workers[0].report_device_id.remote()
    )
    worker_device_id = ray.get(
        vllm_policy.worker_group.workers[0].report_device_id.remote()
    )

    # Ensure they are on the same device
    assert training_device_id == worker_device_id, (
        "Training actor and worker should be on the same device"
    )

    # Use our new utility methods for weight update
    # Get IPC handles from the HF policy
    ipc_handles = hf_policy.get_weights_ipc_handles()
    print("Got IPC handles from HF policy")
    vllm_policy.prepare_for_generation()
    # Update weights in the VllmGeneration
    assert vllm_policy.update_weights(ipc_handles), "Weight update should succeed"

    # Check if weights have been updated
    assert vllm_policy._check_all_weights_changed(), "Weights should be updated to zero"

    # Clean up
    vllm_policy.shutdown()
