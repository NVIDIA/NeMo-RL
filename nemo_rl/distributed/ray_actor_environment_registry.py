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
import collections

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

DEFAULT_ACTOR_ENVIRONMENT = PY_EXECUTABLES.SYSTEM
ACTOR_ENVIRONMENT_REGISTRY = {
    "nemo_rl.models.generation.vllm.VllmGenerationWorker": PY_EXECUTABLES.VLLM,
    "nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker": PY_EXECUTABLES.BASE,
    "nemo_rl.models.policy.dtensor_policy_worker.FSDP1PolicyWorker": PY_EXECUTABLES.BASE,
}
ACTOR_ENVIRONMENT_REGISTRY = collections.defaultdict(
    lambda: DEFAULT_ACTOR_ENVIRONMENT, ACTOR_ENVIRONMENT_REGISTRY
)
