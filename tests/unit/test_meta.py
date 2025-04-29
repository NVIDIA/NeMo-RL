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

# This module tests things outside of any package (e.g., things in the root __init__.py)

import os


def test_usage_stats_disabled_by_default():
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0", (
        "Our dockerfile, slurm submission script and default environment setting when importing nemo rl should all disable usage stats collection. This failing is not expected."
    )


def test_usage_stats_disabled_in_tests():
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0", (
        "Our dockerfile, slurm submission script and default environment setting when importing nemo rl should all disable usage stats collection. This failing is not expected."
    )
