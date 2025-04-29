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
import torch

from nemo_rl.distributed.collectives import (
    gather_jagged_object_lists,
    rebalance_nd_tensor,
)


def run_rebalance_test(rank, world_size):
    """Test function for rebalance_nd_tensor"""
    # Create different sized tensors on each GPU
    # Rank 0: batch size 3, Rank 1: batch size 5, Rank 2: batch size 2
    batch_sizes = [3, 5, 2]
    my_batch_size = batch_sizes[rank]

    tensor = torch.ones(
        (my_batch_size, 4), dtype=torch.float32, device=f"cuda:{rank}"
    ) * (rank + 1)
    result = rebalance_nd_tensor(tensor)

    # Verify the shape is correct (sum of all batch sizes)
    total_batch_size = sum(batch_sizes)
    assert result.shape[0] == total_batch_size, (
        f"Expected shape {total_batch_size}, got {result.shape[0]}"
    )
    assert result.shape[1:] == tensor.shape[1:], "Feature dimensions should match"


def run_gather_test(rank, world_size):
    """Test function for gather_jagged_object_lists"""
    object_lists = [
        ["obj0", "obj1"],  # rank 0: 2 objects
        ["obj2", "obj3", "obj4"],  # rank 1: 3 objects
        ["obj5"],  # rank 2: 1 object
    ]
    my_objects = object_lists[rank]

    result = gather_jagged_object_lists(my_objects)

    expected = ["obj0", "obj1", "obj2", "obj3", "obj4", "obj5"]
    assert len(result) == len(expected), (
        f"Expected {len(expected)} objects, got {len(result)}"
    )
    assert set(result) == set(expected), "All objects should be gathered"


def test_rebalance_nd_tensor(distributed_test_runner):
    """Test rebalance_nd_tensor by spawning multiple processes"""
    distributed_test_runner(run_rebalance_test, world_size=3)


def test_gather_jagged_object_lists(distributed_test_runner):
    """Test gather_jagged_object_lists by spawning multiple processes"""
    distributed_test_runner(run_gather_test, world_size=3)
