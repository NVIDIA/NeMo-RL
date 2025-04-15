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
import ray

from nemo_reinforcer.algorithms.utils import get_tokenizer
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.models.generation.interfaces import configure_generation_config
from nemo_reinforcer.models.generation.vllm import VllmGeneration, VllmConfig
from nemo_reinforcer.models.policy import PolicyConfig


# Define basic vLLM test config
basic_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": "meta-llama/Llama-3.2-1B",  # Small model for testing
    "tokenizer": {
        "name": "meta-llama/Llama-3.2-1B",
    },
    "dtype": "bfloat16",
    "max_new_tokens": 10,
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

# Create HF-specific config with required parameters
basic_hf_test_config: PolicyConfig = {
    "model_name": basic_vllm_test_config["model_name"],
    "tokenizer": {
        "name": basic_vllm_test_config["tokenizer"]["name"],
    },
    # Required training parameters
    "train_global_batch_size": 1,
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
    tokenizer = get_tokenizer(basic_vllm_test_config["tokenizer"])
    return tokenizer


@pytest.fixture(scope="function")
def policy(cluster, tokenizer):
    """Initialize the vLLM policy."""
    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer)
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


def test_vllm_missing_required_config_key(cluster):
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
    vllm_config = configure_generation_config(vllm_config, tokenizer)

    hf_config = basic_hf_test_config.copy()
    hf_config["train_global_batch_size"] = 4

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

        expected_generations = [
            "Write a story about a magical forest. The forest is magical because it is full of",
            "Explain how photosynthesis works\nExplain how photosynthesis works\nPhotosynthesis",
            "What are the benefits of exercise? The benefits of exercise are many and varied. It",
            "Describe the water cycle in your own words.\nDescribe the water cycle in",
            "What is the capital of France? A. Paris B. New York C. Washington",
            "Who is the president of the USA? Who is the president of the USA? Who is",
            "What is the capital of the moon? A. Houston, Texas B. New York City",
            "Where is the sun? Where is the moon? Where is the earth?",
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
        hf_policy = HfPolicy(cluster, hf_config, tokenizer)

        print(f"refitting vllm policy...")
        ipc_handles = hf_policy.get_weights_ipc_handles()
        vllm_policy.prepare_for_generation()
        vllm_policy.update_weights(ipc_handles)

        # Step 1: Use vLLM for generation
        print("Using vLLM policy for fast generation...")
        generation_results = vllm_policy.generate(test_input_data, greedy=True)
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
        assert generated_texts == expected_generations, (
            "Output should be the same as the expected output"
        )

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
    tp_config = deepcopy(basic_vllm_test_config)
    tp_config = configure_generation_config(tp_config, tokenizer)
    tp_config["vllm_cfg"]["tensor_parallel_size"] = 2

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


def test_vllm_generate_text(cluster, tokenizer):
    """Test that vLLM can generate text."""
    # Prepare test data
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    test_prompts = BatchedDataDict({"prompts": test_prompts})

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)

    # Ensure we can get same output
    assert vllm_config["model_name"] == "meta-llama/Llama-3.2-1B", (
        "Model name should be meta-llama/Llama-3.2-1B to get expected output"
    )
    assert vllm_config["vllm_cfg"]["tensor_parallel_size"] == 1, (
        "Tensor parallel size should be 1 to get expected output"
    )

    # Create vLLM generation
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Generate and check result
    output = vllm_generation.generate_text(test_prompts, greedy=True)
    assert output["texts"] == [
        " Kelsey and I am a 2018 graduate",
        " Paris. The city is located in the north of",
    ], "Output should be the same as the expected output"

    # Clean up
    vllm_generation.shutdown()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_vllm_weight_update_and_prefix_cache_reset(
    cluster, tokenizer, tensor_parallel_size
):
    """Test that the vLLM prefix cache is correctly reset when weights change."""
    from nemo_reinforcer.models.policy.hf_policy import HfPolicy

    # Create configs
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
    vllm_config["vllm_cfg"]["tensor_parallel_size"] = tensor_parallel_size
    if tensor_parallel_size > 1:
        vllm_config["vllm_kwargs"] = {"distributed_executor_backend": "ray"}

    hf_config = basic_hf_test_config.copy()

    # Create policies
    vllm_policy = None
    hf_policy = None
    try:
        print(f"Creating HF policy for TP={tensor_parallel_size}...")
        hf_policy = HfPolicy(cluster, hf_config, tokenizer)
        print(f"Creating vLLM policy for TP={tensor_parallel_size}...")
        vllm_policy = VllmGeneration(cluster, vllm_config)

        # Prepare input data (batch size 2)
        text = """Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.Question: What was OKT3 originally sourced from?Answer:"""
        test_prompt = [text, text]  # Use batch size 2
        encodings = tokenizer(
            test_prompt,
            padding=True,
            return_tensors="pt",
            padding_side="right",
        )
        input_ids = encodings["input_ids"]
        input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
        test_input_data = BatchedDataDict(
            {"input_ids": input_ids, "input_lengths": input_lengths}
        )

        print("Running Generation 1 (Initial)...")
        vllm_policy.prepare_for_generation()
        outputs1 = vllm_policy.generate(test_input_data, greedy=True)
        generated_text = tokenizer.decode(
            outputs1["output_ids"][0], skip_special_tokens=True
        )
        print(f"Generated text (Run 1): {generated_text}")
        logprob1 = outputs1["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 1): {logprob1}")

        print("Adding noise to weights in HF policy...")
        ray.get(
            [
                worker._add_noise_to_weights.remote()
                for worker in hf_policy.worker_group.workers
            ]
        )

        print("Updating vLLM weights from HF policy...")
        ipc_handles = hf_policy.get_weights_ipc_handles()
        update_success = vllm_policy.update_weights(ipc_handles)
        assert update_success, "Weight update should succeed"
        print("vLLM weights successfully updated.")

        print("Running Generation 2 (Weights Updated, Cache Still Active)...")
        # Generate again *without* resetting the cache
        outputs2 = vllm_policy.generate(test_input_data, greedy=True)
        logprob2 = outputs2["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 2): {logprob2}")
        assert logprob2 != logprob1, "Logprobs should be different after weight update."

        print("Resetting vLLM prefix cache (via finish/prepare cycle)...")
        vllm_policy.finish_generation()  # Calls sleep() which resets cache
        vllm_policy.prepare_for_generation()  # Calls wake_up()

        print("Running Generation 3 (Weights updated, Cache Reset)...")
        outputs3 = vllm_policy.generate(test_input_data, greedy=True)
        logprob3 = outputs3["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 3): {logprob3}")
        assert logprob2 != logprob3, (
            "Logprobs should be different after cache reset and weight update."
        )

        print("Prefix cache reset verified successfully.")

    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        if vllm_policy:
            vllm_policy.shutdown()
        if hf_policy:
            hf_policy.shutdown()
        # Force garbage collection to help release resources
        import gc

        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("is_eval", [True, False])
def test_vllm_generation_with_stop(cluster, test_input_data, tokenizer, is_eval):
    """Test vLLM generation with stop."""
    from nemo_reinforcer.models.policy.hf_policy import HfPolicy

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config["stop_token_ids"] = [3363]
    vllm_config["stop_strings"] = ["I am a"]
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=is_eval)

    # Ensure we can get same output
    assert vllm_config["model_name"] == "meta-llama/Llama-3.2-1B", (
        "Model name should be meta-llama/Llama-3.2-1B to get expected output"
    )
    assert vllm_config["vllm_cfg"]["tensor_parallel_size"] == 1, (
        "Tensor parallel size should be 1 to get expected output"
    )

    # Create policies
    print("Creating vLLM policy...")
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Get weights from HF policy if not in eval mode
    if not is_eval:
        # set to sleep first if not in eval mode
        vllm_generation.finish_generation()

        print("Creating HF policy...")
        hf_config = basic_hf_test_config.copy()
        hf_policy = HfPolicy(cluster, hf_config, tokenizer)

        print(f"refitting vllm policy...")
        ipc_handles = hf_policy.get_weights_ipc_handles()
        vllm_generation.prepare_for_generation()
        vllm_generation.update_weights(ipc_handles)

    # test generate
    outputs = vllm_generation.generate(test_input_data, greedy=True)
    output_ids = outputs["output_ids"]
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    assert generated_texts == [
        "Hello, my name is Kelsey and I am a",
        "The capital of France is Paris. The city",
    ], "Output should be the same as the expected output"

    # test generate_text
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    test_prompts = BatchedDataDict({"prompts": test_prompts})
    output = vllm_generation.generate_text(test_prompts, greedy=True)
    assert output["texts"] == [
        " Kelsey and I am a",
        " Paris. The city",
    ], "Output should be the same as the expected output"

    # Clean up
    vllm_generation.shutdown()
    if not is_eval:
        hf_policy.shutdown()
