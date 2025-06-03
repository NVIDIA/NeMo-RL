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
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "If you are working interactively, you can install by running  `uv sync --extra vllm` anywhere in the repo."
    )


class VllmInternalWorkerExtension:
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

        with torch.no_grad():
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

            min_layer = min([int(name.split('.')[2]) for name, _ in weights if 'layer' in name])
            max_layer = max([int(name.split('.')[2]) for name, _ in weights if 'layer' in name])
            weights_dict = {name : weight for name, weight in weights}

            from vllm import _custom_ops as ops
            ## fused QKV 
            for layer in range(min_layer, max_layer+1):
                q = weights_dict.pop(f'model.layers.{layer}.self_attn.q_proj.weight')
                k = weights_dict.pop(f'model.layers.{layer}.self_attn.k_proj.weight')
                v = weights_dict.pop(f'model.layers.{layer}.self_attn.v_proj.weight')

                qweight, weight_scale = ops.scaled_fp8_quant(torch.cat((q,k,v)), scale=None)
                qweight = qweight.t()

                self.model_runner.model.model.layers[layer].self_attn.qkv_proj.weight.copy_(qweight)
                self.model_runner.model.model.layers[layer].self_attn.qkv_proj.weight_scale.copy_(weight_scale)

            #fused fc1 and gate MLP
            for layer in range(min_layer, max_layer+1):
                up = weights_dict.pop(f'model.layers.{layer}.mlp.up_proj.weight')
                gate = weights_dict.pop(f'model.layers.{layer}.mlp.gate_proj.weight')
                
                qweight, weight_scale = ops.scaled_fp8_quant(torch.cat((gate, up)), scale=None)
                qweight = qweight.t()

                self.model_runner.model.model.layers[layer].mlp.gate_up_proj.weight.copy_(qweight)
                self.model_runner.model.model.layers[layer].mlp.gate_up_proj.weight_scale.copy_(weight_scale)

            ## RowParallelLinear
            for layer in range(min_layer, max_layer+1):
                o_proj = weights_dict.pop(f'model.layers.{layer}.self_attn.o_proj.weight')
                o_proj_qweight, o_proj_weight_scale = ops.scaled_fp8_quant(o_proj, scale=None)
                o_proj_qweight = o_proj_qweight.t()

                down_proj = weights_dict.pop(f'model.layers.{layer}.mlp.down_proj.weight')
                down_proj_qweight, down_proj_weight_scale = ops.scaled_fp8_quant(down_proj, scale=None)
                down_proj_qweight = down_proj_qweight.t()

                self.model_runner.model.model.layers[layer].self_attn.o_proj.weight.copy_(o_proj_qweight)
                self.model_runner.model.model.layers[layer].self_attn.o_proj.weight_scale.copy_(o_proj_weight_scale)
                self.model_runner.model.model.layers[layer].mlp.down_proj.weight.copy_(down_proj_qweight)
                self.model_runner.model.model.layers[layer].mlp.down_proj.weight_scale.copy_(down_proj_weight_scale)

            weights = [(k,v) for k,v in weights_dict.items()]
            self.model_runner.model.load_weights(weights=weights)

        torch.cuda.synchronize()
        
        return True 

