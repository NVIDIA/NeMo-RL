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
from nemo_reinforcer.distributed.virtual_cluster import (
    _get_node_ip_and_free_port,
    PY_EXECUTABLES,
    RayVirtualCluster,
    ResourceInsufficientError,
)
import ray
import pytest
import os
from unittest.mock import patch, MagicMock
import importlib


def test_get_node_ip_and_free_port_does_not_start_with_zero():
    # This test covers a case where the hostname was an integer like "255"
    # and socket returned an ip address equivalent to this hostname, i.e., "0.0.0.255".
    # It's not possible to mock the way the hostname is actually set on other platforms,
    # so we leave this test so we can ask users to run on their environment if needed.

    node_ip, _ = ray.get(
        _get_node_ip_and_free_port.options(
            runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
        ).remote()
    )
    assert not node_ip.startswith("0."), "Node IP should not start with 0.*.*.*"


def test_env_max_retries_invalid_value():
    """Test that NRL_VIRTUAL_CLUSTER_MAX_RETRIES rejects invalid values (less than or equal to zero)."""

    # Mock environment with invalid max_retries value
    env_vars = {"NRL_VIRTUAL_CLUSTER_MAX_RETRIES": "0"}

    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(AssertionError):
            RayVirtualCluster(bundle_ct_per_node_list=[1])


def test_env_max_retries_non_integer():
    """Test that NRL_VIRTUAL_CLUSTER_MAX_RETRIES handles non-integer values properly."""

    # Mock environment with non-integer max_retries value
    env_vars = {"NRL_VIRTUAL_CLUSTER_MAX_RETRIES": "not_a_number"}

    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValueError):
            RayVirtualCluster(bundle_ct_per_node_list=[1])


def test_env_max_retries_default_value():
    """Test that default value for NRL_VIRTUAL_CLUSTER_MAX_RETRIES is used when not set."""

    # Ensure environment variable is not set
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "nemo_reinforcer.distributed.virtual_cluster.RayVirtualCluster._init_placement_groups"
        ) as mock_init,
    ):
        # Mock successful initialization
        mock_init.return_value = [MagicMock()]

        # Create cluster
        cluster = RayVirtualCluster(bundle_ct_per_node_list=[1])

        # Default value should be 6 (as seen in the code)
        # We can't directly verify this, but we can check that initialization was attempted
        assert mock_init.call_count == 1


def test_env_max_retries_exhausted():
    """Test that NRL_VIRTUAL_CLUSTER_MAX_RETRIES correctly handles the case where all retries fail."""

    # Set specific retry count to 4
    retry_count = 4
    env_vars = {"NRL_VIRTUAL_CLUSTER_MAX_RETRIES": str(retry_count)}

    with (
        patch.dict(os.environ, env_vars, clear=True),
        patch(
            "nemo_reinforcer.distributed.virtual_cluster.RayVirtualCluster._init_placement_groups"
        ) as mock_init,
        patch("time.sleep") as mock_sleep,
    ):
        # Make _init_placement_groups raise ResourceInsufficientError each time
        mock_init.side_effect = ResourceInsufficientError("Not enough resources")

        # Create cluster - should retry retry_count times and then fail
        with pytest.raises(ResourceInsufficientError):
            RayVirtualCluster(bundle_ct_per_node_list=[1])

        # Verify _init_placement_groups was called exactly retry_count times
        assert mock_init.call_count == retry_count

        # Verify time.sleep was called with exponentially increasing values
        assert mock_sleep.call_count == retry_count
        mock_sleep.assert_any_call(1)  # 2^0
        mock_sleep.assert_any_call(2)  # 2^1
        mock_sleep.assert_any_call(4)  # 2^2
        mock_sleep.assert_any_call(8)  # 2^3
