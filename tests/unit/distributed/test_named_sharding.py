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
import numpy as np

from nemo_reinforcer.distributed.named_sharding import NamedSharding


@pytest.fixture
def sample_sharding():
    """Provides a standard NamedSharding instance for testing."""
    layout = [[[0, 1, 2, 3], [4, 5, 6, 7]]]  # dp=1, pp=2, tp=4
    names = ["dp", "pp", "tp"]
    return NamedSharding(layout, names)


def test_initialization_success(sample_sharding):
    assert sample_sharding.shape == {"dp": 1, "pp": 2, "tp": 4}
    assert sample_sharding.names == ["dp", "pp", "tp"]
    assert sample_sharding.ndim == 3
    assert sample_sharding.size == 8
    np.testing.assert_array_equal(
        sample_sharding.layout, np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]])
    )


def test_initialization_dim_mismatch():
    layout = [[0, 1], [2, 3]]
    names = ["dp", "pp", "tp"]
    with pytest.raises(ValueError, match="Number of dimensions.*must match"):
        NamedSharding(layout, names)


def test_initialization_non_integer():
    layout = [[0, 1.5], [2, 3]]
    names = ["dp", "pp"]
    with pytest.raises(ValueError, match="Layout must contain only integer rank IDs"):
        NamedSharding(layout, names)


def test_initialization_duplicate_ranks():
    layout = [[0, 1], [2, 0]]
    names = ["dp", "pp"]
    with pytest.raises(ValueError, match="Duplicate ranks found"):
        NamedSharding(layout, names)


def test_get_ranks_full_slice(sample_sharding):
    # Get all ranks for dp=0
    ranks = sample_sharding.get_ranks(dp=0)
    correct_out = NamedSharding(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), ["pp", "tp"])
    assert ranks == correct_out


def test_get_ranks_partial_slice(sample_sharding):
    # Get ranks for dp=0, pp=1
    ranks = sample_sharding.get_ranks(dp=0, pp=1)
    correct_out = NamedSharding(np.array([4, 5, 6, 7]), ["tp"])
    assert ranks == correct_out


def test_get_ranks_partial_slice_2(sample_sharding):
    ranks = sample_sharding.get_ranks(dp=0, tp=2)
    correct_out = NamedSharding(np.array([2, 6]), ["pp"])
    assert ranks == correct_out


def test_get_ranks_single_rank(sample_sharding):
    # Get rank for dp=0, pp=0, tp=2
    ranks = sample_sharding.get_ranks(dp=0, pp=0, tp=2)
    correct_out = 2
    assert ranks == correct_out


def test_get_ranks_no_args(sample_sharding):
    # Get all ranks flattened
    ranks = sample_sharding.get_ranks()
    assert ranks == sample_sharding


def test_get_ranks_invalid_name(sample_sharding):
    with pytest.raises(ValueError, match="Invalid axis name: 'xx'"):
        sample_sharding.get_ranks(xx=0)


def test_get_ranks_index_out_of_bounds(sample_sharding):
    with pytest.raises(IndexError, match="Index 2 is out of bounds for axis 'pp'"):
        sample_sharding.get_ranks(pp=2)
    with pytest.raises(IndexError, match="Index 4 is out of bounds for axis 'tp'"):
        sample_sharding.get_ranks(tp=4)


def test_get_axis_index(sample_sharding):
    assert sample_sharding.get_axis_index("dp") == 0
    assert sample_sharding.get_axis_index("pp") == 1
    assert sample_sharding.get_axis_index("tp") == 2


def test_get_axis_index_invalid_name(sample_sharding):
    with pytest.raises(ValueError, match="Invalid axis name: 'xx'"):
        sample_sharding.get_axis_index("xx")


def test_get_axis_size(sample_sharding):
    assert sample_sharding.get_axis_size("dp") == 1
    assert sample_sharding.get_axis_size("pp") == 2
    assert sample_sharding.get_axis_size("tp") == 4


def test_equality():
    layout1 = [[[0, 1], [2, 3]]]
    names1 = ["a", "b", "c"]
    sharding1 = NamedSharding(layout1, names1)

    layout2 = [[[0, 1], [2, 3]]]
    names2 = ["a", "b", "c"]
    sharding2 = NamedSharding(layout2, names2)

    layout3 = [[[0, 1], [2, 4]]]  # Different layout
    names3 = ["a", "b", "c"]
    sharding3 = NamedSharding(layout3, names3)

    layout4 = [[[0, 1], [2, 3]]]
    names4 = ["x", "y", "z"]  # Different names
    sharding4 = NamedSharding(layout4, names4)

    assert sharding1 == sharding2
    assert sharding1 != sharding3
    assert sharding1 != sharding4
    assert sharding1 != "not a sharding object"


def test_repr(sample_sharding):
    representation = repr(sample_sharding)
    assert "NamedSharding" in representation
    assert "shape=(1, 2, 4)" in representation
    assert "names=['dp', 'pp', 'tp']" in representation
    assert "layout=" in representation
    assert "[[[0 1 2 3]" in representation  # Check layout content part
    assert "[4 5 6 7]]]" in representation
