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

import importlib
import os
import torch
from contextlib import contextmanager
from copy import deepcopy


def import_class_from_path(name):
    """Import a class from a string path (e.g. 'torch.optim.AdamW').

    Args:
        full_path: Full path to class including module path and class name

    Returns:
        The imported class object
    """
    module_name, cls_name = name.rsplit(".", 1)
    cls_instance = getattr(importlib.import_module(module_name), cls_name)
    return cls_instance


def get_gpu_info(model):
    """Return information about the GPU being used by this worker."""
    import torch

    # Get distributed training info
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get device info from CUDA
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # in MB
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # in MB

    # Try to get the real global device ID (not the local one)
    # In distributed training, each process only sees its assigned GPU as device 0
    local_device_id = device
    global_device_id = local_device_id

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if local_rank < len(cuda_visible_devices):
            global_device_id = int(cuda_visible_devices[local_rank])

    # Get a parameter from the model to verify CUDA device placement
    # This confirms tensors are actually on the appropriate device
    param_info = {}
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None and param.requires_grad:
                full_name = f"{module_name}.{param_name}"
                param_info[full_name] = {
                    "device": str(param.device),
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                }
                # Just grab one parameter for verification
                break
        if param_info:
            break

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "local_device_id": local_device_id,
        "global_device_id": global_device_id,
        "device_count": device_count,
        "device_name": device_name,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "peak_memory_allocated_mb": peak_memory,
        "peak_memory_reserved_mb": peak_reserved,
        "parameter_sample": param_info,
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
        },
    }


def convert_to_amp_o2_format(state_dict):
    """When amp_o2 is enabled, the model gets wrapped in a Float16Module which changes the keys and how it loads need to add module onto it."""
    new_state_dict = {}

    for key, item in state_dict.items():
        if "model.module." not in key:
            key = key.replace("model.", "model.module.", 1)
        new_state_dict[key] = item

    return new_state_dict


def retrieve_model_state_dict_in_cpu(model, megatron_amp_O2=True):
    """Get a copy of the model states in CPU."""
    cpu_dict = {}

    for name, item in model.state_dict().items():
        if isinstance(item, torch.Tensor):
            item = item.detach().to(device="cpu", non_blocking=True, copy=True)

        cpu_dict[name] = item

    if megatron_amp_O2:
        cpu_dict = convert_to_amp_o2_format(cpu_dict)

    torch.cuda.synchronize()
    return cpu_dict


@torch.no_grad()
def swap_dict(resident_model, cpu_weights, offload_onto_cpu=True, megatron_amp_O2=True):
    """Swap the state dict with a specified state dict, and offload the current state dict onto CPU if needed."""
    offloaded_weights = {}

    if offload_onto_cpu:
        offloaded_weights = retrieve_model_state_dict_in_cpu(
            resident_model, megatron_amp_O2=megatron_amp_O2
        )

    resident_model.load_state_dict(cpu_weights)
    return offloaded_weights


@contextmanager
def cpu_weight_swap(resident_model, cpu_weights, megatron_amp_O2=True):
    """Swap the weights into GPU, and then swap it out once return."""
    cpu_dict = swap_dict(resident_model, cpu_weights, megatron_amp_O2=megatron_amp_O2)
    try:
        yield

    finally:
        swap_dict(
            resident_model,
            cpu_dict,
            offload_onto_cpu=False,
            megatron_amp_O2=megatron_amp_O2,
        )


@torch.no_grad()
def copy_model_states_to_cpu(
    model, cpu_dict=None, megatron_amp_O2=True, sync=True, alias_non_tensor=False
):
    """Mutates the cpu_dict object to throw the model states into preallocated tensors(if they exist).

    For non tensors it will do a deepcopy, unless alias_non_tensor is True.
    """
    if cpu_dict is None:
        cpu_dict = {}

    for name, item in model.state_dict().items():
        if isinstance(item, torch.Tensor):
            if name not in cpu_dict:
                cpu_dict[name] = torch.empty(
                    item.size(),
                    dtype=item.dtype,
                    layout=item.layout,
                    device="cpu",
                    pin_memory=True,
                )
            cpu_dict[name].copy_(item, non_blocking=sync)
        elif alias_non_tensor:
            cpu_dict[name] = item
        else:
            cpu_dict[name] = deepcopy(item)

    if megatron_amp_O2:
        cpu_dict = convert_to_amp_o2_format(cpu_dict)

    if sync:
        torch.cuda.synchronize()

    return cpu_dict
