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
import sys
from pathlib import Path

"""
This is a work around to ensure whenever NeMo RL is imported, that we
add Megatron-LM to the python path. This is because the only sub-package
that's officially installed is megatron.core. So we add the whole repo into
the path so we can access megatron.{training,legacy,inference,...}

Since users may pip install NeMo RL, this is a convenience so they do not
have to manually run with PYTHONPATH=3rdparty/Megatron-LM-workspace/Megatron-LM.
"""
megatron_path = (
    Path(__file__).parent.parent / "3rdparty" / "Megatron-LM-workspace" / "Megatron-LM"
)
if megatron_path.exists() and str(megatron_path) not in sys.path:
    sys.path.append(str(megatron_path))

from nemo_rl.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)

os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
