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

from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster


@pytest.fixture(autouse=True)
def mock_virtual_cluster_pg():
    # Mock the _init_placement_groups and get_placement_groups methods to avoid actually initializing placement groups
    with (
        patch(
            "nemo_rl.distributed.virtual_cluster.RayVirtualCluster.get_placement_groups"
        ) as mock_get_pg,
        patch(
            "nemo_rl.distributed.virtual_cluster.RayVirtualCluster._init_placement_groups"
        ) as mock_init_pg,
    ):
        mock_get_pg.return_value = []
        mock_init_pg.return_value = []
        yield


def test_empty_cluster_visualization(capsys):
    """Test visualization of an empty cluster."""
    # Create a empty cluster
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[],
        use_gpus=False,
        name="test-empty",
    )

    # Test visualization
    cluster.print_cluster_grid()

    # Capture the output
    out, _ = capsys.readouterr()
    assert "Empty Ray Cluster" in out


def test_cluster_grid(capsys):
    """Test visualization of a cluster grid."""
    # Create a cluster with a configuration but don't actually allocate resources
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2, 3],
        use_gpus=False,
        name="test-visual",
        max_colocated_worker_groups=1,
    )

    cluster.print_cluster_grid()

    # Capture the output
    out, _ = capsys.readouterr()
    print(out)
    assert "Ray Cluster: 2 nodes, 5 GPUs" in out
    assert "0.0" in out  # First node, first GPU
    assert "0.1" in out  # First node, second GPU
    assert "1.0" in out  # Second node, first GPU
    assert "1.2" in out  # Second node, third GPU


def test_global_visualization_formatting(capsys):
    """Test global visualization formatting without actual worker groups."""
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2, 2],
        use_gpus=False,
        name="test-global",
        max_colocated_worker_groups=1,
    )

    cluster.print_all_worker_groups([])

    # Capture the output
    out, _ = capsys.readouterr()
    print(out)
    assert "Ray Cluster Global View: 2 nodes, 4 GPUs" in out


def test_with_mock_worker_groups(capsys):
    """Test visualization with mock worker groups."""
    # Create a cluster with a configuration
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2, 3],
        use_gpus=False,
        name="test-workers",
        max_colocated_worker_groups=1,
    )

    worker_group1 = MagicMock()
    worker_group1.name_prefix = "policy"
    worker_group1.world_size = 2
    worker_group1.worker_metadata = [
        {"node_idx": 0, "local_rank": 0},  # First worker on node 0, GPU 0
        {"node_idx": 1, "local_rank": 0},  # Second worker on node 1, GPU 0
    ]

    worker_group2 = MagicMock()
    worker_group2.name_prefix = "policy_generate"
    worker_group2.world_size = 3
    worker_group2.worker_metadata = [
        {"node_idx": 0, "local_rank": 1},  # First worker on node 0, GPU 1
        {"node_idx": 1, "local_rank": 1},  # Second worker on node 1, GPU 1
        {"node_idx": 1, "local_rank": 2},  # Third worker on node 1, GPU 2
    ]

    cluster.print_all_worker_groups([worker_group1, worker_group2])

    # Capture the output
    out, _ = capsys.readouterr()
    print(out)

    # Check for key elements in the output
    assert "Ray Cluster Global View: 2 nodes, 5 GPUs" in out
    assert "G0" in out  # First worker group
    assert "G1" in out  # Second worker group
    assert "policy" in out  # First worker group name
    assert "policy_generate" in out  # Second worker group name
