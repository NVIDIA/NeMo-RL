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
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict


def test_shard_by_batch_size_basic():
    """Test basic functionality of shard_by_batch_size with tensor data."""
    # Create a sample batch with tensor data
    batch = BatchedDataDict(
        {
            "tensor_data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            "other_tensor": torch.tensor([10, 11, 12, 13, 14, 15, 16, 17]),
        }
    )

    # Shard with batch_size=4, shards=2
    sharded = batch.shard_by_batch_size(shards=2, batch_size=4)

    # Verify output structure
    assert len(sharded) == 2, f"Expected 2 shards, got {len(sharded)}"

    # Verify first shard content (first elements of each chunk)
    assert torch.equal(sharded[0]["tensor_data"], torch.tensor([0, 1, 4, 5]))
    assert torch.equal(sharded[0]["other_tensor"], torch.tensor([10, 11, 14, 15]))

    # Verify second shard content (second elements of each chunk)
    assert torch.equal(sharded[1]["tensor_data"], torch.tensor([2, 3, 6, 7]))
    assert torch.equal(sharded[1]["other_tensor"], torch.tensor([12, 13, 16, 17]))


def test_shard_by_batch_size_list_data():
    """Test shard_by_batch_size with list data."""
    # Create a sample batch with list data
    batch = BatchedDataDict(
        {
            "list_data": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "tensor_data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        }
    )

    # Shard with batch_size=4, shards=2
    sharded = batch.shard_by_batch_size(shards=2, batch_size=4)

    # Verify output structure
    assert len(sharded) == 2

    # Verify first shard content
    assert sharded[0]["list_data"] == ["A", "B", "E", "F"]
    assert torch.equal(sharded[0]["tensor_data"], torch.tensor([0, 1, 4, 5]))

    # Verify second shard content
    assert sharded[1]["list_data"] == ["C", "D", "G", "H"]
    assert torch.equal(sharded[1]["tensor_data"], torch.tensor([2, 3, 6, 7]))


def test_shard_by_batch_size_larger_example():
    """Test shard_by_batch_size with a larger example with multiple chunks and shards."""
    # Create a batch with 12 elements
    batch = BatchedDataDict(
        {"tensor_data": torch.arange(12), "list_data": [f"item_{i}" for i in range(12)]}
    )

    # Shard with batch_size=3, shards=3
    sharded = batch.shard_by_batch_size(shards=3, batch_size=3)

    # Verify we get 3 shards
    assert len(sharded) == 3

    # Expected results:
    # Chunk 1: [0, 1, 2], Chunk 2: [3, 4, 5], Chunk 3: [6, 7, 8], Chunk 4: [9, 10, 11]
    # Shard 1: [0, 3, 6, 9]
    # Shard 2: [1, 4, 7, 10]
    # Shard 3: [2, 5, 8, 11]

    # Verify tensor content
    assert torch.equal(sharded[0]["tensor_data"], torch.tensor([0, 3, 6, 9]))
    assert torch.equal(sharded[1]["tensor_data"], torch.tensor([1, 4, 7, 10]))
    assert torch.equal(sharded[2]["tensor_data"], torch.tensor([2, 5, 8, 11]))

    # Verify list content
    assert sharded[0]["list_data"] == ["item_0", "item_3", "item_6", "item_9"]
    assert sharded[1]["list_data"] == ["item_1", "item_4", "item_7", "item_10"]
    assert sharded[2]["list_data"] == ["item_2", "item_5", "item_8", "item_11"]


def test_shard_by_batch_size_2d_tensor():
    """Test shard_by_batch_size with 2D tensor data."""
    # Create a batch with 2D tensors
    batch = BatchedDataDict(
        {
            "features": torch.tensor(
                [
                    [1, 2, 3],  # 0
                    [4, 5, 6],  # 1
                    [7, 8, 9],  # 2
                    [10, 11, 12],  # 3
                    [13, 14, 15],  # 4
                    [16, 17, 18],  # 5
                ]
            )
        }
    )

    # Shard with batch_size=3, shards=3
    sharded = batch.shard_by_batch_size(shards=3, batch_size=3)

    # Verify we get 3 shards
    assert len(sharded) == 3

    # Expected results by index:
    # Chunk 1: [0, 1, 2], Chunk 2: [3, 4, 5]
    # Shard 1: [0, 3]
    # Shard 2: [1, 4]
    # Shard 3: [2, 5]

    # Verify tensor content
    expected_0 = torch.tensor([[1, 2, 3], [10, 11, 12]])
    expected_1 = torch.tensor([[4, 5, 6], [13, 14, 15]])
    expected_2 = torch.tensor([[7, 8, 9], [16, 17, 18]])

    assert torch.equal(sharded[0]["features"], expected_0)
    assert torch.equal(sharded[1]["features"], expected_1)
    assert torch.equal(sharded[2]["features"], expected_2)


def test_shard_by_batch_size_edge_cases():
    """Test edge cases for shard_by_batch_size."""
    # Case 1: Single batch, multiple shards
    batch = BatchedDataDict({"data": torch.tensor([0, 1, 2, 3])})

    sharded = batch.shard_by_batch_size(shards=2, batch_size=4)
    assert len(sharded) == 2
    assert torch.equal(sharded[0]["data"], torch.tensor([0, 1]))
    assert torch.equal(sharded[1]["data"], torch.tensor([2, 3]))

    # Case 2: Multiple batches, single shard
    batch = BatchedDataDict({"data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])})

    sharded = batch.shard_by_batch_size(shards=1, batch_size=2)
    assert len(sharded) == 1
    assert torch.equal(sharded[0]["data"], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))


def test_shard_by_batch_size_validation():
    """Test validation checks in shard_by_batch_size."""
    # Create a batch
    batch = BatchedDataDict({"data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])})

    # Case 1: batch_size not a divisor of total_batch_size
    with pytest.raises(
        AssertionError, match="Total batch size.*is not a multiple of batch_size"
    ):
        batch.shard_by_batch_size(shards=2, batch_size=3)

    # Case 2: shards not a divisor of batch_size
    # First make a batch that's divisible by batch_size to reach the second assertion
    batch_for_case2 = BatchedDataDict({"data": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])})
    with pytest.raises(AssertionError, match="Batch size.*is not a multiple of shards"):
        batch_for_case2.shard_by_batch_size(shards=3, batch_size=4)

    # Case 3: Different batch sizes across keys
    inconsistent_batch = BatchedDataDict(
        {
            "data1": torch.tensor([0, 1, 2, 3]),
            "data2": torch.tensor([0, 1, 2]),
        }  # Different length
    )

    with pytest.raises(
        AssertionError, match="Batch sizes are not the same across the rollout batch"
    ):
        inconsistent_batch.shard_by_batch_size(shards=2, batch_size=2)


def test_shard_by_batch_size_matches_example():
    """Test that shard_by_batch_size behaves as described in the docstring example."""
    # Create the example data: [A A B B C C D D]
    batch = BatchedDataDict({"data": ["A", "A", "B", "B", "C", "C", "D", "D"]})

    # Shard with batch_size=2, shards=2
    sharded = batch.shard_by_batch_size(shards=2, batch_size=2)

    # Verify output structure
    assert len(sharded) == 2

    # Expected output:
    # Element 0: [A B C D] (first elements from each chunk)
    # Element 1: [A B C D] (second elements from each chunk)
    assert sharded[0]["data"] == ["A", "B", "C", "D"]
    assert sharded[1]["data"] == ["A", "B", "C", "D"]


def test_make_microbatch_iterator_perfect_division():
    """Test make_microbatch_iterator when batch size is perfectly divisible by microbatch_size."""
    # Create a batch with 8 elements (perfectly divisible by microbatch_size=2)
    batch = BatchedDataDict(
        {"tensor_data": torch.arange(8), "list_data": [f"item_{i}" for i in range(8)]}
    )

    # Create microbatch iterator with microbatch_size=2
    iterator = batch.make_microbatch_iterator(max_microbatch_size=2)
    microbatches = list(iterator)

    # Verify we get exactly 4 microbatches
    assert len(microbatches) == 4, f"Expected 4 microbatches, got {len(microbatches)}"

    # Verify contents of each microbatch
    for i, microbatch in enumerate(microbatches):
        start_idx = i * 2
        end_idx = start_idx + 2
        assert torch.equal(microbatch["tensor_data"], torch.arange(start_idx, end_idx))
        assert microbatch["list_data"] == [
            f"item_{j}" for j in range(start_idx, end_idx)
        ]

    # Also test with microbatch_size=4
    iterator = batch.make_microbatch_iterator(max_microbatch_size=4)
    microbatches = list(iterator)

    # Verify we get exactly 2 microbatches
    assert len(microbatches) == 2

    # Verify contents
    assert torch.equal(microbatches[0]["tensor_data"], torch.arange(0, 4))
    assert torch.equal(microbatches[1]["tensor_data"], torch.arange(4, 8))
    assert microbatches[0]["list_data"] == ["item_0", "item_1", "item_2", "item_3"]
    assert microbatches[1]["list_data"] == ["item_4", "item_5", "item_6", "item_7"]


def test_make_microbatch_iterator_smaller_than_microbatch():
    """Test make_microbatch_iterator when batch size is smaller than microbatch_size."""
    # Create a batch with 3 elements (smaller than microbatch_size=5)
    batch = BatchedDataDict(
        {"tensor_data": torch.arange(3), "list_data": ["A", "B", "C"]}
    )

    # Create microbatch iterator with microbatch_size=5 (larger than batch)
    iterator = batch.make_microbatch_iterator(max_microbatch_size=5)
    microbatches = list(iterator)

    # Verify we get exactly 1 microbatch containing the entire batch
    assert len(microbatches) == 1
    assert torch.equal(microbatches[0]["tensor_data"], torch.arange(3))
    assert microbatches[0]["list_data"] == ["A", "B", "C"]

    # Edge case: batch size 0
    empty_batch = BatchedDataDict({"tensor_data": torch.tensor([]), "list_data": []})
    iterator = empty_batch.make_microbatch_iterator(max_microbatch_size=3)
    microbatches = list(iterator)

    # Should either return an empty list or a single empty microbatch
    # (assuming the implementation handles empty batches)
    if microbatches:
        assert len(microbatches) == 1
        assert len(microbatches[0]["tensor_data"]) == 0
        assert len(microbatches[0]["list_data"]) == 0


def test_make_microbatch_iterator_with_leftovers():
    """Test make_microbatch_iterator with leftovers in the final microbatch."""
    # Create a batch with 11 elements (not perfectly divisible by microbatch_size=4)
    batch = BatchedDataDict(
        {
            "tensor_data": torch.arange(11),
            "list_data": [f"item_{i}" for i in range(11)],
            "dict_data": [{"id": i} for i in range(11)],
        }
    )

    # Create microbatch iterator with microbatch_size=4
    iterator = batch.make_microbatch_iterator(max_microbatch_size=4)
    microbatches = list(iterator)

    # Verify we get 3 microbatches (2 full, 1 with leftovers)
    assert len(microbatches) == 3

    # Verify first two microbatches have size 4
    assert len(microbatches[0]["tensor_data"]) == 4
    assert len(microbatches[1]["tensor_data"]) == 4

    # Verify last microbatch has the leftover 3 elements
    assert len(microbatches[2]["tensor_data"]) == 3

    # Verify contents
    assert torch.equal(microbatches[0]["tensor_data"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(microbatches[1]["tensor_data"], torch.tensor([4, 5, 6, 7]))
    assert torch.equal(microbatches[2]["tensor_data"], torch.tensor([8, 9, 10]))

    assert microbatches[0]["list_data"] == ["item_0", "item_1", "item_2", "item_3"]
    assert microbatches[1]["list_data"] == ["item_4", "item_5", "item_6", "item_7"]
    assert microbatches[2]["list_data"] == ["item_8", "item_9", "item_10"]

    assert microbatches[0]["dict_data"] == [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]
    assert microbatches[1]["dict_data"] == [{"id": 4}, {"id": 5}, {"id": 6}, {"id": 7}]
    assert microbatches[2]["dict_data"] == [{"id": 8}, {"id": 9}, {"id": 10}]

    # Test with a different microbatch_size that produces a small leftover
    iterator = batch.make_microbatch_iterator(max_microbatch_size=5)
    microbatches = list(iterator)

    # Verify we get 3 microbatches (2 full, 1 with single leftover)
    assert len(microbatches) == 3
    assert len(microbatches[0]["tensor_data"]) == 5
    assert len(microbatches[1]["tensor_data"]) == 5
    assert len(microbatches[2]["tensor_data"]) == 1

    assert torch.equal(microbatches[2]["tensor_data"], torch.tensor([10]))
    assert microbatches[2]["list_data"] == ["item_10"]
    assert microbatches[2]["dict_data"] == [{"id": 10}]
