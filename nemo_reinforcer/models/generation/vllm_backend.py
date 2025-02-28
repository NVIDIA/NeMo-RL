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

try:
    from vllm.worker.worker import Worker
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please install it with `pip install nemo-reinforcer[vllm]` "
        "or `pip install vllm` separately. This issue may also occur if worker is using incorrect "
        "py_executable."
    )


class UpdatableVllmInternalWorker(Worker):
    def report_device_id(self) -> str:
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

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
            for name, handle in handles.items():
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
                f"Error in UpdatableVllmInternalWorker.update_weights_from_ipc_handles: {e}"
            )
            return False

    def check_weights_changed(self):
        """Check if the weights are updated to 0.

        Returns:
            bool: True if all weights have been zeroed, False otherwise.
        """
        try:
            weights_updated = True
            for name, p in self.model_runner.model.named_parameters():
                weights_updated = weights_updated and torch.allclose(
                    p, torch.zeros_like(p)
                )
            return weights_updated
        except Exception as e:
            print(f"Error in UpdatableVllmInternalWorker.check_weights_changed: {e}")
            return False
