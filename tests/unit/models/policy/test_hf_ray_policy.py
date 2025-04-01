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
import ray
import pytest
import pprint
import torch

from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.algorithms.interfaces import LossFunction
from tests.unit.test_utils import simple_loss, nll_loss
from transformers import AutoTokenizer


basic_llama_test_config: PolicyConfig = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "generation_batch_size": 1,  # Small batch size for testing
    "train_global_batch_size": 4,
    "train_micro_batch_size": 1,
    "learning_rate": 5e-6,
    "logprob_batch_size": 1,
    "precision": "float32",
    "generation": {
        "backend": "hf",
        "temperature": 1.0,
        "max_new_tokens": 16,  # Small number of tokens for testing
        "top_p": 1.0,
        "top_k": None,
    },
    "optimizer": {
        "name": "torch.optim.AdamW",
        "kwargs": {
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    },
    "scheduler": {
        "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "kwargs": {
            "T_max": 100,
        },
    },
}


@pytest.fixture(scope="function")
def gc_collect():
    """Helper function to force garbage collection after a test"""
    import gc

    gc.collect()


@pytest.fixture
def policy_setup():
    """Setup and teardown for policy tests - creates a virtual cluster and policy."""
    policy = None
    cluster = None

    cluster_name = "test"
    print(f"Creating virtual cluster '{cluster_name}'...")

    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[2],  # Single node, 2 gpus
        use_gpus=True,
        num_gpus_per_node=2,  # Using both GPUs
        max_colocated_worker_groups=1,  # Only one worker group
    )

    config = basic_llama_test_config

    print("Creating HfPolicy...")
    policy = HfPolicy(cluster=cluster, config=config)

    yield policy, cluster

    # Clean up after the test
    print("Cleaning up resources for test")
    cluster.shutdown()
    policy.worker_group.shutdown()


@pytest.mark.timeout(180)
def test_hf_policy_init(policy_setup):
    policy, cluster = policy_setup

    # Verify cluster and policy were properly created
    assert policy is not None, "Policy was not created properly"
    assert cluster is not None, "Cluster was not created properly"

    # Verify we have two workers, one per GPU
    assert len(policy.worker_group.workers) == 2, "Should have 2 workers, one per GPU"

    # Check workers are alive
    worker_alive = ray.get([w.is_alive.remote() for w in policy.worker_group.workers])
    assert all(worker_alive), f"Not all workers are alive: {worker_alive}"

    # Get GPU info from both workers to verify GPU usage
    print("\nGetting GPU information from workers...")
    gpu_infos = ray.get([w.get_gpu_info.remote() for w in policy.worker_group.workers])
    print("\nGPU Information:")
    for i, info in enumerate(gpu_infos):
        print(f"\nWorker {i} GPU Info:")
        pprint.pprint(info)

    # Check 1: Verify workers have different ranks
    gpu_ranks = [info["rank"] for info in gpu_infos]
    assert len(set(gpu_ranks)) == 2, f"Expected 2 different ranks, got {gpu_ranks}"
    assert set(gpu_ranks) == {0, 1}, f"Expected ranks 0 and 1, got {gpu_ranks}"

    # Check 2: Verify workers have different local_ranks
    local_ranks = [info["local_rank"] for info in gpu_infos]
    assert len(set(local_ranks)) == 2, (
        f"Expected 2 different local_ranks, got {local_ranks}"
    )
    assert set(local_ranks) == {0, 1}, (
        f"Expected local_ranks 0 and 1, got {local_ranks}"
    )

    # Check 3: Verify workers have different CUDA_VISIBLE_DEVICES
    cuda_visible_devices = [
        info["env_vars"].get("CUDA_VISIBLE_DEVICES") for info in gpu_infos
    ]
    assert len(set(cuda_visible_devices)) == 2, (
        f"Expected different CUDA_VISIBLE_DEVICES, got {cuda_visible_devices}"
    )

    # Check 4: Verify all workers report correct world_size
    for info in gpu_infos:
        assert info["world_size"] == 2, (
            f"Expected world_size=2, got {info['world_size']}"
        )
        assert info["env_vars"]["WORLD_SIZE"] == "2", (
            f"Expected WORLD_SIZE=2, got {info['env_vars']['WORLD_SIZE']}"
        )

    # Check 5: Verify significant GPU memory is allocated (at least 1GB) on both GPUs
    for info in gpu_infos:
        assert info["memory_allocated_mb"] > 1000, (
            f"Not enough memory allocated on GPU for rank {info['rank']}: {info['memory_allocated_mb']:.2f} MB"
        )

    # Check 6: Verify model parameters are on CUDA devices for both workers
    for info in gpu_infos:
        param_sample = list(info["parameter_sample"].values())[0]
        assert "cuda" in param_sample["device"], (
            f"Parameter not on CUDA device: {param_sample['device']}"
        )

    # Check 8: Verify same model parameters are being tracked across workers
    param_names = [list(info["parameter_sample"].keys())[0] for info in gpu_infos]
    assert len(set(param_names)) == 1, (
        f"Workers are not tracking the same parameter: {param_names}"
    )

    # Check 9: Both workers should see their device as cuda:0 (correct distributed behavior)
    for info in gpu_infos:
        param_device = list(info["parameter_sample"].values())[0]["device"]
        assert param_device == "cuda:0", (
            f"Expected parameter device to be cuda:0, got {param_device}"
        )


@pytest.fixture
def training_setup():
    """Setup and teardown specifically for training tests."""
    policy = None
    cluster = None
    data = None
    loss_fn = None

    try:
        # Create resources with unique name
        cluster_name = "test-train"
        print(f"Creating training virtual cluster '{cluster_name}'...")

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[2],  # Single node, 2 gpus
            use_gpus=True,
            num_gpus_per_node=2,  # Using both GPUs
            max_colocated_worker_groups=1,  # Only one worker group
        )

        config = basic_llama_test_config

        print("Creating training HfPolicy...")
        policy = HfPolicy(cluster=cluster, config=config, init_reference_model=False)

        # Create a test batch
        print("Creating test batch...")
        # set random seed
        torch.manual_seed(42)

        # Create test input_ids and attention_mask
        input_ids = torch.randint(0, 32000, (8, 128))  # 8 sequences, each of length 128
        attention_mask = torch.ones(8, 128)

        # Calculate input_lengths (all sequences are full length in this test)
        input_lengths = attention_mask.sum(dim=1).to(torch.int32)

        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,  # Keep for compatibility with loss functions
                "labels": torch.randint(0, 32000, (8, 128)),
            }
        )

        # Create loss function
        loss_fn: LossFunction = simple_loss

        # Provide the resources to the test
        yield policy, cluster, data, loss_fn

    except Exception as e:
        print(f"Error during training setup: {e}")
        pytest.skip(f"Training setup failed: {e}")
    finally:
        # Clean up after the test
        print("Cleaning up resources for test")
        cluster.shutdown()
        policy.worker_group.shutdown()


@pytest.mark.timeout(180)
def test_hf_policy_training(training_setup):
    def verify_loss_tensor(loss_tensor):
        assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
        assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
        return loss_tensor

    policy, cluster, data, loss_fn = training_setup

    # Verify resources were created properly
    assert policy is not None, "Training policy was not created properly"
    assert cluster is not None, "Training cluster was not created properly"
    assert data is not None, "Test data was not created properly"
    assert loss_fn is not None, "Loss function was not created properly"

    # Call prepare_for_training if available
    print("\nPreparing for training...")
    policy.prepare_for_training()

    losses = []
    for steps in range(4):
        results = policy.train(data, loss_fn)

        # Verify results
        assert "loss" in results, "Training results should contain 'loss'"
        loss_tensor = results["loss"]
        verify_loss_tensor(loss_tensor)
        losses.append(loss_tensor[-1].item())

        print(f"Training loss: {results['loss']}")

    policy.finish_training()

    # Verify loss changed between iterations (model parameters were updated)
    assert losses[0] > losses[-1], "Loss should decrease over training iterations"


@pytest.fixture
def generation_setup(request):
    """Setup and teardown specifically for generation tests."""
    policy = None
    cluster = None
    data = None

    try:
        # Create resources with unique name
        cluster_name = "test-generate"
        print(f"Creating generation virtual cluster '{cluster_name}'...")

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[2],  # Single node, 2 gpus
            use_gpus=True,
            num_gpus_per_node=2,  # Using both GPUs
            max_colocated_worker_groups=1,  # Only one worker group
        )

        config = basic_llama_test_config

        print("Creating generation HfPolicy...")
        policy = HfPolicy(
            cluster=cluster, config=config, init_reference_model=request.param
        )

        # Create a test batch
        print("Creating test batch...")
        torch.manual_seed(42)  # For reproducibility

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
            "Write a story about a magical forest. The forest is magical because it is full of magical creatures. The creatures are",
            "Explain how photosynthesis works\nExplain how photosynthesis works\nPhotosynthesis is the process by which plants",
            "What are the benefits of exercise? The benefits of exercise are many and varied. It is a great way to improve",
            "Describe the water cycle in your own words.\nDescribe the water cycle in your own words.\nDescribe the",
            "What is the capital of France? A. Paris B. New York C. Washington D. Baton Rouge\nA",
            "Who is the president of the USA? Who is the president of the USA? Who is the president of the USA?",
            "What is the capital of the moon? A. Houston B. New York C. Washington D. Denver\nA.",
            "Where is the sun? Where is the moon? Where is the earth? Where is the sky? Where",
        ]

        # Tokenize the prompts
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        tokenizer.pad_token = tokenizer.eos_token
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

        data = BatchedDataDict(
            {
                "input_ids": tokenized["input_ids"],
                "input_lengths": input_lengths,
            }
        )

        # Provide the resources to the test
        yield policy, cluster, data, tokenizer, prompts, expected_generations

    except Exception as e:
        print(f"Error during generation setup: {e}")
        pytest.skip(f"Generation setup failed: {e}")
    finally:
        # Clean up after the test
        print("Cleaning up resources for test")
        cluster.shutdown()
        policy.worker_group.shutdown()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("generation_setup", [False], indirect=True)
def test_hf_policy_generation(generation_setup, tracker):
    policy, cluster, data, tokenizer, prompts, expected_generations = generation_setup

    # Verify resources were created properly
    assert policy is not None, "Generation policy was not created properly"
    assert cluster is not None, "Generation cluster was not created properly"
    assert data is not None, "Test data was not created properly"

    # Call prepare_for_generation if available
    print("Preparing for generation...")
    policy.prepare_for_generation()

    # Generate text
    print("Generating text...")
    results = policy.generate(data, greedy=True)

    # Verify results
    assert "output_ids" in results, "Generation results should contain 'output_ids'"
    output_ids = results["output_ids"]

    # run logprob calculation manually to verify
    fprop_logprob_data = BatchedDataDict(
        {
            "input_ids": results.get("output_ids"),
            "input_lengths": results.get("unpadded_sequence_lengths"),
        }
    )
    fprop_results = policy.get_logprobs(fprop_logprob_data)
    for i, length in enumerate(data["input_lengths"]):
        fprop_results["logprobs"][i, :length] = 0

    for i, valid_seq_len in enumerate(results["unpadded_sequence_lengths"]):
        fprop_results["logprobs"][i, valid_seq_len:] = 0

    # Basic validation of output shape and content
    assert isinstance(output_ids, torch.Tensor), "Output should be a tensor"
    assert output_ids.dim() == 2, (
        "Output should be 2-dimensional [batch_size, seq_length]"
    )
    assert output_ids.size(0) == data.get("input_ids").size(0), (
        "Output batch size should match input"
    )
    assert output_ids.size(1) > data.get("input_ids").size(1), (
        "Output should be longer than input"
    )

    # validate that the logprobs are correct
    avg_prob_mult_error = torch.mean(
        torch.exp(torch.abs(results["logprobs"] - fprop_results["logprobs"]))
    )
    print(f"avg prob mult error: {avg_prob_mult_error}")
    tracker.track("avg_prob_mult_error", float(avg_prob_mult_error))
    assert avg_prob_mult_error <= 1.025

    # get logprobs for the expected generations
    expected_tokenized = tokenizer(
        expected_generations,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input_lengths for expected generations
    expected_lengths = expected_tokenized["attention_mask"].sum(dim=1).to(torch.int32)

    expected_data = BatchedDataDict(
        {
            "input_ids": expected_tokenized["input_ids"],
            "input_lengths": expected_lengths,
        }
    )

    expected_logprobs = policy.get_logprobs(expected_data)["logprobs"]
    mean_lps = torch.mean(expected_logprobs * expected_tokenized["attention_mask"])
    tracker.track("mean_lps", float(mean_lps))
    assert mean_lps > -1.7, "Expected logprobs should be greater than -1.7"
    assert mean_lps < -1.4, "Expected logprobs should be less than -1.4"

    # Call finish_generation if available
    print("Finishing generation...")
    policy.finish_generation()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("generation_setup", [True], indirect=True)
def test_all_hf_policy_generation_lps_ref_training(generation_setup):
    policy, cluster, data, tokenizer, prompts, expected_generations = generation_setup

    # Verify resources were created properly
    assert policy is not None, "Generation policy was not created properly"
    assert cluster is not None, "Generation cluster was not created properly"
    assert data is not None, "Test data was not created properly"

    # Create reference data by generating with the model
    print("creating some data")
    ref_results = policy.generate(data, greedy=True)

    # Create training data with reference outputs
    token_loss_mask = torch.ones_like(ref_results["output_ids"])
    token_loss_mask[:, : data.get("input_ids").size(1)] = 0

    for idx, length in enumerate(ref_results["unpadded_sequence_lengths"]):
        token_loss_mask[idx, length:] = 0

    train_data = BatchedDataDict(
        {
            "input_ids": ref_results["output_ids"],
            "input_lengths": ref_results["unpadded_sequence_lengths"],
            "token_loss_mask": token_loss_mask,
        }
    )

    fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]

    loss_fn: LossFunction = nll_loss

    # Train for a few steps
    policy.prepare_for_training()
    losses = []
    for step in range(8):
        results = policy.train(train_data, loss_fn)

        # Verify results
        assert "loss" in results, "Training results should contain 'loss'"
        loss_tensor = results["loss"]
        assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
        assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
        losses.append(loss_tensor[-1].item())

        print(f"Training loss at step {step}: {results['loss']}")

    policy.finish_training()

    post_train_reference_logprobs = policy.get_reference_policy_logprobs(train_data)[
        "reference_logprobs"
    ]
    post_train_fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]

    # Verify that the reference policy logprobs match the original policy logprobs
    assert torch.allclose(fprop_logprobs, post_train_reference_logprobs), (
        "Logprobs from policy before training and reference policy after training should match"
    )

    # Calculate NLL before and after training
    pre_train_nll = -torch.sum(fprop_logprobs * token_loss_mask, dim=-1)
    post_train_nll = -torch.sum(post_train_fprop_logprobs * token_loss_mask, dim=-1)
    print(f"Pre-training NLL: {pre_train_nll.mean().item()}")
    print(f"Post-training NLL: {post_train_nll.mean().item()}")

    # Verify that training improved the model's predictions on every sample
    assert torch.all(post_train_nll < pre_train_nll), (
        "Model should improve at predicting its own generations after training"
    )
    assert torch.all(post_train_nll < 5), (
        "Model should improve at predicting its own generations after training"
    )

    # Verify loss decreased during training
    assert losses[0] > losses[-1], "Loss should decrease over training iterations"
