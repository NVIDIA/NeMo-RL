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
import ray
from nemo_reinforcer.environments.math_environment import MathEnvironment
import time
import os


@pytest.fixture(scope="module")
def math_env():
    """Create a MathEnvironment actor for testing."""
    env = MathEnvironment.options(
        runtime_env={
            "py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),
        }
    ).remote({"num_workers": 2})
    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


@pytest.fixture
def basic_test_data():
    """Common test data for basic math problems."""
    return {
        "message_log_batch": [
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = \\boxed{4}"},
            ],
            [
                {"role": "user", "content": "What is 3 * 4?"},
                {"role": "assistant", "content": "3 * 4 = \\boxed{12}"},
            ],
            [
                {"role": "user", "content": "What is 10 - 5?"},
                {"role": "assistant", "content": "10 - 5 = \\boxed{5}"},
            ],
        ],
        "metadata": [
            {"ground_truth": "4"},
            {"ground_truth": "\\boxed{12}"},
            {"ground_truth": "\\boxed{5}"},
        ],
    }


@pytest.fixture
def mixed_test_data():
    """Test data with mix of correct and incorrect responses."""
    return {
        "message_log_batch": [
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = \\boxed{\\frac{8}{2}}"},
            ],
            [
                {"role": "user", "content": "What is 3 * 4?"},
                {"role": "assistant", "content": "3 * 4 = 13"},
            ],
            [
                {"role": "user", "content": "What is 10 - 5?"},
                {"role": "assistant", "content": "10 - 5 = \\boxed{5}"},
            ],
        ],
        "metadata": [
            {"ground_truth": "4.0"},
            {"ground_truth": "\\boxed{12}"},
            {"ground_truth": "\\boxed{5}"},
        ],
    }


@pytest.fixture
def multiple_assistant_test_data():
    """Test data with multiple assistant messages in conversations."""
    return {
        "message_log_batch": [
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "Let me think..."},
                {"role": "assistant", "content": "2 + 2 = \\boxed{4}"},
            ],
            [
                {"role": "user", "content": "What is 3 * 4?"},
                {"role": "assistant", "content": "I'll calculate that..."},
                {"role": "assistant", "content": "3 * 4 = \\boxed{12}"},
            ],
        ],
        "metadata": [{"ground_truth": "4"}, {"ground_truth": "\\boxed{12}"}],
    }


def test_math_env_step_basic(math_env, basic_test_data):
    """Test basic functionality of MathEnvironment step with simple messages."""
    observations, updated_metadata, rewards, done = ray.get(
        math_env.step.remote(
            basic_test_data["message_log_batch"], basic_test_data["metadata"]
        )
    )

    # Check observations
    assert len(observations) == 3, "Should return observations for all 3 messages"
    assert all(obs["role"] == "user" for obs in observations), (
        "All observations should be from user"
    )
    assert all(obs["content"] == "correct" for obs in observations), (
        "All responses should be correct"
    )

    # Check metadata
    assert len(updated_metadata) == 3, "Should return metadata for all 3 messages"
    assert updated_metadata == basic_test_data["metadata"], (
        "Metadata should be unchanged"
    )

    # Check rewards and done flags
    assert rewards.shape == (3,), "Rewards should be a tensor of shape (3,)"
    assert all(rewards == 1.0), "All rewards should be 1.0 for correct answers"
    assert done.shape == (3,), "Done flags should be a tensor of shape (3,)"
    assert all(done == 1.0), "All done flags should be 1.0"


def test_math_env_step_mixed(math_env, mixed_test_data):
    """Test MathEnvironment step with a mix of correct and incorrect responses."""
    observations, updated_metadata, rewards, done = ray.get(
        math_env.step.remote(
            mixed_test_data["message_log_batch"], mixed_test_data["metadata"]
        )
    )

    # Check observations and rewards
    assert len(observations) == 3, "Should return observations for all 3 messages"
    assert observations[0]["content"] == "correct", "First response should be correct"
    assert observations[1]["content"] == "incorrect", (
        "Second response should be incorrect"
    )
    assert observations[2]["content"] == "correct", "Third response should be correct"

    assert rewards.shape == (3,), "Rewards should be a tensor of shape (3,)"
    assert rewards[0] == 1.0, "First reward should be 1.0"
    assert rewards[1] == 0.0, "Second reward should be 0.0"
    assert rewards[2] == 1.0, "Third reward should be 1.0"


def test_math_env_step_empty(math_env):
    """Test MathEnvironment step with empty input."""
    observations, updated_metadata, rewards, done = ray.get(
        math_env.step.remote([], [])
    )

    # Check all outputs are empty
    assert len(observations) == 0, "Should return empty observations list"
    assert len(updated_metadata) == 0, "Should return empty metadata list"
    assert rewards.shape == (0,), "Should return empty rewards tensor"
    assert done.shape == (0,), "Should return empty done tensor"


def test_math_env_step_multiple_assistant_messages(
    math_env, multiple_assistant_test_data
):
    """Test MathEnvironment step with multiple assistant messages in a conversation."""
    observations, updated_metadata, rewards, done = ray.get(
        math_env.step.remote(
            multiple_assistant_test_data["message_log_batch"],
            multiple_assistant_test_data["metadata"],
        )
    )

    # Check that only the last assistant message is used
    assert len(observations) == 2, "Should return observations for both conversations"
    assert all(obs["content"] == "correct" for obs in observations), (
        "All responses should be correct"
    )
    assert all(rewards == 1.0), "All rewards should be 1.0"


@pytest.mark.parametrize("batch_size", [1, 2, 10, 25, 101])
def test_math_env_various_batches(math_env, batch_size):
    """Test MathEnvironment step with different batch sizes."""
    message_log_batch = [
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 1.333 = \\boxed{\\frac{10}{3}}"},
        ]
    ] * batch_size
    metadata = [{"ground_truth": "3.33333333"}] * batch_size

    observations, updated_metadata, rewards, done = ray.get(
        math_env.step.remote(message_log_batch, metadata)
    )

    # Check outputs
    assert len(observations) == batch_size, (
        f"Should return observations for all {batch_size} messages"
    )
    assert all(obs["content"] == "correct" for obs in observations), (
        "All responses should be correct"
    )
    assert all(rewards == 1.0), "All rewards should be 1.0"
    assert all(done == 1.0), "All done flags should be 1.0"
