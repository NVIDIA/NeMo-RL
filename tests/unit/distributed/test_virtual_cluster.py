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
)
import ray


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
