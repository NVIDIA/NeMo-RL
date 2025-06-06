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
from typing import Any

import torch

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "If you are working interactively, you can install by running  `uv sync --extra vllm` anywhere in the repo."
    )


class VllmInternalWorkerExtension:
    def init_collective(self, rank_prefix: int, world_size: int) -> None:
        """Initialize the collective communication."""
        import ray.util.collective as collective

        local_rank = torch.distributed.get_rank()
        rank = rank_prefix + local_rank + 1  # 1 is the head node of the train cluster

        collective.init_collective_group(
            world_size=world_size, rank=rank, backend="nccl", group_name="refit"
        )

    def report_device_id(self) -> str:
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def update_weights_from_ipc_handles(self, ipc_handles):
        """Update weights from IPC handles.

        Args:
            ipc_handles (dict): Dictionary mapping device UUIDs to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated.
        """
        try:
            # Get handles for this device
            device_uuid = self.report_device_id()
            handles = ipc_handles[device_uuid]
            device_id = self.device.index
            weights = []

            # Process each handle to get the tensor
            for name, handle in handles:
                func, args = handle
                list_args = list(args)
                # Update device ID to match the current device
                list_args[6] = device_id
                tensor = func(*list_args)
                weights.append((name, tensor))

            # Load weights into the model
            self.model_runner.model.load_weights(weights=weights)
            torch.cuda.synchronize()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_ipc_handles: {e}"
            )
            return False

    def update_weights_from_collective(self, info: dict[str, Any]) -> bool:
        """Update the model weights from collective communication."""
        import ray.util.collective as collective

        try:
            for name, (shape, dtype) in info.items():
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                collective.broadcast(weight, 0, group_name="refit")
                self.model_runner.model.load_weights(weights=[(name, weight)])
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True
