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

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.utils import (
    get_logprobs,
    get_tokenizer,
    log_metrics,
    reduce_microbatch_metrics,
    save_checkpoint,
    setup_checkpointer,
    setup_dataloaders,
    validate_checkpointing_config,
)
from nemo_rl.data.hf_datasets.chat_templates import COMMON_CHAT_TEMPLATES
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.checkpoint import CheckpointManager


@pytest.fixture
def conversation_messages():
    """Fixture providing a multi-turn conversation for testing chat templates"""
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's the weather like today?"},
        {
            "role": "assistant",
            "content": "I don't have access to real-time weather data.",
        },
        {"role": "user", "content": "Can you help me with something else then?"},
        {"role": "assistant", "content": "Of course! What would you like help with?"},
    ]


def get_expected_llama_format(messages):
    """Generate the expected output format for Llama's chat template"""
    # Extract the date from the formatted output
    # Get current date
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d %b %Y")

    # Extract system message if present
    if messages[0]["role"] == "system":
        system_message = messages[0]["content"].strip()
        messages = messages[1:]
    else:
        system_message = ""

    # Start with BOS token and system header
    expected = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    expected += "Cutting Knowledge Date: December 2023\n"
    expected += f"Today Date: {formatted_date}\n\n"
    expected += f"{system_message}<|eot_id|>"

    # Add each message
    for message in messages:
        if message["role"] not in ["ipython", "tool"]:
            expected += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            expected += f"{message['content'].strip()}<|eot_id|>"

    return expected


def get_format_with_simple_role_header(messages):
    message = "<|begin_of_text|>"
    for msg in messages:
        message += (
            "<|start_header_id|>"
            + msg["role"]
            + "<|end_header_id|>\n\n"
            + msg["content"].strip()
            + "<|eot_id|>"
        )
    return message


def test_get_tokenizer_no_chat_template(conversation_messages):
    """Test get_tokenizer when no chat template is specified in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct"}
    tokenizer = get_tokenizer(config)

    # Verify that the tokenizer's default template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)

    expected = get_expected_llama_format(conversation_messages)
    assert formatted == expected


def test_get_tokenizer_default_chat_template(conversation_messages):
    """Test get_tokenizer when chat_template is 'default' in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct", "chat_template": "default"}
    tokenizer = get_tokenizer(config)

    # Verify that the tokenizer's default template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)
    expected = get_expected_llama_format(conversation_messages)
    assert formatted == expected


def test_get_tokenizer_null_chat_template(conversation_messages):
    """Test get_tokenizer when chat_template is None in config"""
    config = {"name": "meta-llama/Llama-3.2-1B-Instruct", "chat_template": None}
    tokenizer = get_tokenizer(config)

    # Verify that the passthrough template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)

    expected = "".join(msg["content"] for msg in conversation_messages)

    assert formatted == expected


def test_get_tokenizer_custom_jinja_template(conversation_messages):
    """Test get_tokenizer when a custom jinja template is specified"""
    custom_template = COMMON_CHAT_TEMPLATES.simple_role_header
    config = {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "chat_template": custom_template,
    }
    tokenizer = get_tokenizer(config)

    # Verify that the custom template is used
    formatted = tokenizer.apply_chat_template(conversation_messages, tokenize=False)
    expected = get_format_with_simple_role_header(conversation_messages)
    assert formatted == expected


@pytest.fixture
def mock_checkpointer_config():
    return {
        "enabled": True,
        "save_period": 100,
        "checkpoint_dir": "test_checkpoints",
        "keep_top_k": 3,
        "metric_name": "validation_loss",
        "higher_is_better": False,  # For loss metrics, lower is better
    }


@pytest.fixture
def mock_algorithm_config():
    return {
        "val_period": 50,
        "val_at_start": True,
        "seed": 42,
        "val_batch_size": 8,
    }


@pytest.fixture
def mock_policy_config():
    return {
        "train_global_batch_size": 16,
    }


def test_setup_checkpointer(mock_checkpointer_config):
    default_save_state = {"step": 0, "epoch": 0}

    with patch("nemo_rl.utils.checkpoint.CheckpointManager") as mock_checkpoint_manager:
        mock_checkpoint_manager.return_value.get_latest_checkpoint_path.return_value = (
            None
        )

        checkpointer, last_checkpoint_path, save_state = setup_checkpointer(
            mock_checkpointer_config, default_save_state
        )

        assert isinstance(checkpointer, CheckpointManager)
        assert last_checkpoint_path is None
        assert save_state == default_save_state


def test_validate_checkpointing_config(mock_checkpointer_config, mock_algorithm_config):
    # Test valid config
    validate_checkpointing_config(mock_checkpointer_config, mock_algorithm_config)

    # Test invalid config
    invalid_config = mock_checkpointer_config.copy()
    invalid_config["save_period"] = 75  # Not a multiple of val_period (50)

    with pytest.raises(AssertionError):
        validate_checkpointing_config(invalid_config, mock_algorithm_config)


def test_setup_dataloaders(mock_algorithm_config, mock_policy_config):
    # Create mock datasets
    train_dataset = MagicMock()
    train_dataset.__len__.return_value = 100
    val_dataset = MagicMock()
    val_dataset.__len__.return_value = 20

    def mock_collate_fn(batch):
        return batch

    train_dataloader, val_dataloader = setup_dataloaders(
        train_dataset,
        val_dataset,
        mock_collate_fn,
        mock_algorithm_config,
        mock_policy_config,
    )

    assert isinstance(train_dataloader, StatefulDataLoader)
    assert isinstance(val_dataloader, StatefulDataLoader)
    assert train_dataloader.batch_size == mock_policy_config["train_global_batch_size"]
    assert val_dataloader.batch_size == mock_algorithm_config["val_batch_size"]


def test_save_checkpoint(mock_checkpointer_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_checkpointer_config["checkpoint_dir"] = tmpdir
        checkpointer = CheckpointManager(mock_checkpointer_config)
        checkpointer.finalize_checkpoint = MagicMock()

        master_config = {"model": "test"}
        save_state = {"step": 100}
        total_steps = 100
        train_dataloader = MagicMock()
        policy = MagicMock()
        timer = MagicMock()
        timer.time.return_value.__enter__.return_value = None

        def mock_save(obj, path):
            # Create an empty file at the specified path
            with open(path, "w") as f:
                f.write("")

        with patch("torch.save", side_effect=mock_save):
            save_checkpoint(
                checkpointer,
                master_config,
                save_state,
                total_steps,
                train_dataloader,
                policy,
                timer,
            )

            # Verify policy.save_checkpoint was called
            policy.save_checkpoint.assert_called_once()
            # Verify train_dataloader state was saved in the tmp location
            # in practice, this will be the actual checkpoint location,
            # but we mocked the finalize_checkpoint fn to do nothing to avoid
            # making this test too complex
            assert os.path.exists(
                os.path.join(tmpdir, "tmp_step_100/train_dataloader.pt")
            )


def test_reduce_microbatch_metrics():
    metrics = {
        "global_valid_seqs": [1, 2, 3],
        "global_valid_toks": [4, 5, 6],
        "loss": [0.1, 0.2, 0.3],
        "accuracy": [0.8, 0.9, 1.0],
    }

    reduced = reduce_microbatch_metrics(metrics)

    assert abs(reduced["global_valid_seqs"] - 2) < 1e-5  # Mean
    assert abs(reduced["global_valid_toks"] - 5) < 1e-5  # Mean
    assert abs(reduced["loss"] - 0.6) < 1e-5  # Sum
    assert abs(reduced["accuracy"] - 2.7) < 1e-5  # Sum


def test_log_metrics():
    log_to_console = {"loss": 0.5, "accuracy": 0.9}
    metrics = {"loss": 0.5, "accuracy": 0.9, "other_metric": 0.7}
    timing_metrics = {
        "total_step_time": 1.0,
        "total_validation_time": 2.0,
        "forward_time": 0.3,
        "backward_time": 0.7,
    }
    step = 100
    logger = MagicMock()

    # Test training metrics
    log_metrics(log_to_console, metrics, timing_metrics, step, logger, is_val=False)
    logger.log_metrics.assert_any_call(metrics, step, prefix="train")
    logger.log_metrics.assert_any_call(timing_metrics, step, prefix="timing/train")

    # Test validation metrics
    logger.reset_mock()
    log_metrics(log_to_console, metrics, timing_metrics, step, logger, is_val=True)
    logger.log_metrics.assert_any_call(metrics, step, prefix="validation")
    logger.log_metrics.assert_any_call(timing_metrics, step, prefix="timing/validation")


def test_get_logprobs():
    """Test the get_logprobs function for both regular tensors and DTensors."""
    import torch

    # Test case 1: Regular tensor
    batch_size = 2
    seq_len = 4
    vocab_size = 5

    # Create input data
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device="cuda")
    data = BatchedDataDict({"input_ids": input_ids})

    # Create logits (batch_size, seq_len, vocab_size)
    # Use deterministic values for easier testing
    next_token_logits = torch.zeros(batch_size, seq_len, vocab_size, device="cuda")
    # Set high logits for the tokens we want to predict
    for b in range(batch_size):
        for s in range(seq_len - 1):
            next_token_logits[b, s, input_ids[b, s + 1]] = 10.0

    # Get log probabilities
    logprobs = get_logprobs(data, next_token_logits)

    # Verify shape and device
    assert logprobs.shape == (
        batch_size,
        seq_len - 1,
    )  # -1 because we remove last position
    assert logprobs.device.type == "cuda"

    # Verify values are log probabilities (should be negative)
    assert torch.all(logprobs <= 0)

    # Since we set high logits for the correct tokens, their logprobs should be close to 0
    expected_tokens = input_ids[:, 1:]  # Skip first token
    for b in range(batch_size):
        for s in range(seq_len - 1):
            token = expected_tokens[b, s]
            assert logprobs[b, s] > -0.003

    # Test case 2: DTensor
    # Skip DTensor test if distributed training is not set up
    if not torch.distributed.is_available():
        print("Skipping DTensor test: torch.distributed is not available")
        return

    try:
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor

        # Set up distributed environment variables
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        # Initialize process group
        torch.distributed.init_process_group(backend="nccl")

        # Create a simple device mesh
        device_mesh = DeviceMesh("cuda", [0])

        # Create DTensor logits
        dtensor_logits = DTensor.from_local(
            next_token_logits,
            device_mesh,
        )

        # Get log probabilities with DTensor
        dtensor_logprobs = get_logprobs(data, dtensor_logits)

        # Verify shape and device
        assert dtensor_logprobs.shape == (batch_size, seq_len - 1)
        assert dtensor_logprobs.device.type == "cuda"

        # Verify DTensor results match regular tensor results
        assert torch.allclose(logprobs, dtensor_logprobs)

        # Clean up distributed environment
        torch.distributed.destroy_process_group()

    except (ImportError, RuntimeError) as e:
        # Skip DTensor test if not available or if distributed setup fails
        print(f"Skipping DTensor test: {str(e)}")
