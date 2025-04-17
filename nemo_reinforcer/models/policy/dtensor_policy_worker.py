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

import gc

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Optional

import ray
import torch
from torch.distributed.fsdp import (
    FSDPModule,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemo_reinforcer.models.dtensor.parallelize import _parallelize_model

from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.policy.utils import import_class_from_path
from nemo_reinforcer.distributed.virtual_cluster import (
    PY_EXECUTABLES,
)
from typing import Iterable, Tuple, Union
from torch.distributed.tensor import DTensor
from nemo_reinforcer.models.dtensor.parallelize import (
    get_logprobs_from_vocab_parallel_logits,
    get_grad_norm,
    clip_grad_by_total_norm_,
    to_local_if_dtensor,
)
from nemo_reinforcer.utils.native_checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from torch import nn
from nemo_reinforcer.models.policy.utils import get_gpu_info


@contextmanager
def unshard_fsdp2_model(model: nn.Module):
    """Explicitly unshard and then reshard the FSDP2 modules. Useful for logprob inference."""
    try:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()


@torch.no_grad()
def get_cpu_state_dict(
    state_generator: Iterable[Tuple[str, Union[torch.Tensor, DTensor]]],
    pin_memory: bool = False,
) -> Dict[str, torch.Tensor]:
    """Copy the state dict generator to CPU memory.

    Args:
        state_generator (Iterable[Tuple[str, Union[torch.Tensor, DTensor]]]):
            An iterable that yields (key, tensor) pairs from a model state.
        pin_memory (bool, optional):
            Whether to allocate the CPU tensors in pinned memory for faster GPU transfer.
            Defaults to False.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping parameter names to CPU tensors.
    """
    new_state_dict = {}
    for k, v in state_generator:
        val = to_local_if_dtensor(v)

        cpu_tensor = torch.empty(
            *val.shape, device="cpu", pin_memory=pin_memory, dtype=val.dtype
        )
        cpu_tensor.copy_(val, non_blocking=True)
        new_state_dict[k] = cpu_tensor

    torch.cuda.synchronize()
    return new_state_dict


@ray.remote
class DTensorPolicyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.BASE

    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: AutoTokenizer,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
    ):
        self.cfg = config
        # torch distributed init. Envars for rank, world_size, and master_addr and master_port are set from the ray remote call
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_name = self.cfg["model_name"]

        self.cpu_offload = self.cfg["dtensor_cfg"]["cpu_offload"]
        self.max_grad_norm = self.cfg["max_grad_norm"]

        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")

        print(f"[Rank {rank}] Loading model {model_name} on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # load weights onto CPU initially
            torch_dtype=torch.float32,  # use full precision in sft until https://github.com/NVIDIA/reinforcer/issues/13 is fixed
        )
        self.tokenizer = tokenizer
        # ------------------------------------------------
        # 3) Move to GPU + Composable FSDP
        #    (Initialize device mesh, shard submodules, then shard entire model)
        # ------------------------------------------------

        tp_size = self.cfg["dtensor_cfg"]["tensor_parallel_size"]
        dp_size = world_size // tp_size
        assert world_size % tp_size == 0, (
            f"World size({world_size}) must be divisible by TP size({tp_size}) to use DTensor"
        )

        mesh_2d = torch.distributed.device_mesh.init_device_mesh(
            "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        self.dp_mesh, self.tp_mesh = mesh_2d["dp"], mesh_2d["tp"]
        self.dp_size = dp_size
        self.tp_size = tp_size

        self.model = _parallelize_model(
            self.model,
            self.dp_mesh,
            self.tp_mesh,
            param_dtype=self.dtype,
            sequence_parallel=self.cfg["dtensor_cfg"]["sequence_parallel"],
            cpu_offload=self.cpu_offload,
            activation_checkpointing=self.cfg["dtensor_cfg"][
                "activation_checkpointing"
            ],
        )

        if self.cpu_offload:
            self.model = self.move_buffer_to_device(self.model, "cpu")

        self._held_model_params = None

        if init_reference_model:
            self.reference_model_state_dict = get_cpu_state_dict(
                self.model.state_dict().items(), pin_memory=True
            )
            self.reference_model_buffers = get_cpu_state_dict(
                self.model.named_buffers(), pin_memory=True
            )

        if init_optimizer:
            optimizer_cls = import_class_from_path(self.cfg["optimizer"]["name"])
            self.optimizer = optimizer_cls(
                self.model.parameters(), **self.cfg["optimizer"]["kwargs"]
            )
        else:
            self.optimizer = None

        if "scheduler" in self.cfg and self.optimizer is not None:
            if isinstance(self.cfg["scheduler"], dict):
                scheduler_cls = import_class_from_path(self.cfg["scheduler"]["name"])
                self.scheduler = scheduler_cls(
                    self.optimizer, **self.cfg["scheduler"]["kwargs"]
                )
            else:
                schedulers = []
                for scheduler_cfg in self.cfg["scheduler"]:
                    if "name" in scheduler_cfg:
                        schedulers.append(
                            import_class_from_path(scheduler_cfg["name"])(
                                self.optimizer, **scheduler_cfg["kwargs"]
                            )
                        )
                    else:
                        assert "milestones" in scheduler_cfg, (
                            "unknown scheduler config: ",
                            scheduler_cfg,
                        )
                        milestones = scheduler_cfg["milestones"]

                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers, milestones
                )

        elif self.optimizer is not None:
            ## default to a passthrough LR schedule
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )

        # restore
        if weights_path:
            self.load_checkpoint(weights_path, optimizer_path)
        else:
            print(
                "No weights path provided. Starting from scratch (default policy init)"
            )

    def is_alive(self):
        return True

    def get_gpu_info(self):
        """Return information about the GPU being used by this worker."""
        return get_gpu_info(self.model)

    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        dataset_size = data.get("input_ids").shape[0]

        # Ensure model is in training mode
        self.model.train()

        # Get data from batch and move to device
        data.to("cuda")

        losses = []
        all_mb_metrics = []
        for gb_start in range(0, dataset_size, local_gbs):
            self.optimizer.zero_grad()
            mb_losses = []

            # Calculate number of microbatches to process
            # make_microbatch_iterator assumes that the batch size is a multiple of the microbatch size
            # so its safe to not check for the case where the last data slice is smaller than mbs
            num_microbatches = min(local_gbs, dataset_size - gb_start) // mbs

            for mb in data.slice(
                gb_start, gb_start + local_gbs
            ).make_microbatch_iterator(mbs):
                input_ids = mb.get("input_ids").cuda()

                input_lengths = mb.get("input_lengths")
                batch_size, seq_len = input_ids.shape

                attention_mask = torch.zeros(
                    (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                )
                for i, length in enumerate(input_lengths):
                    # For right-padded sequence, set 1s at the beginning of the sequence
                    attention_mask[i, :length] = 1

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    batch_size, seq_len = input_ids.shape

                    attention_mask_input_all_ones = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )
                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_input_all_ones,
                        position_ids=position_ids,
                        use_cache=False,
                    )

                # Get logprobs
                if not hasattr(outputs, "logits"):
                    logits = self.model.lm_head(outputs.last_hidden_state)
                else:
                    logits = outputs.logits

                loss, loss_metrics = loss_fn(logits, mb)
                loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                # Backward pass

                # Loss is accumulated across microbatches so we need to scale by the number of microbatches
                loss = loss / num_microbatches
                if not eval_mode:
                    loss.backward()
                mb_losses.append(loss.item())
                all_mb_metrics.append(loss_metrics)

            grad_norm = None
            if not eval_mode:
                with torch.no_grad():
                    grad_norm = get_grad_norm(
                        self.model.parameters(),
                        dp_group=self.dp_mesh.get_group(),
                        tp_group=self.tp_mesh.get_group(),
                        dtype=torch.float32,
                    )
                    if self.max_grad_norm is not None:
                        clip_grad_by_total_norm_(
                            self.model.parameters(),
                            max_grad_norm=self.max_grad_norm,
                            total_norm=grad_norm,
                            dtype=torch.float32,
                        )

                # Update parameters
                self.optimizer.step()
                self.scheduler.step()

            losses.append(torch.tensor(mb_losses).sum().item())

        # Compute global loss across all ranks
        with torch.no_grad():
            local_loss = torch.tensor(losses, device="cuda")
            global_loss = torch.zeros_like(local_loss)
            torch.distributed.all_reduce(local_loss)
            global_loss = local_loss / (self.dp_size * self.tp_size)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        metrics = {
            "global_loss": global_loss.cpu(),
            "local_loss": local_loss.cpu(),
            "grad_norm": grad_norm,
            "rank": torch.distributed.get_rank(),
            "all_mb_metrics": dict(mb_metrics),
        }

        return metrics

    def get_logprobs(self, data: BatchedDataDict) -> BatchedDataDict:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.

        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        logprob_batch_size = self.cfg["logprob_batch_size"]
        all_log_probs = []
        self.model.eval()

        with unshard_fsdp2_model(self.model), torch.no_grad():
            data.to("cuda")
            for lp_batch in data.make_microbatch_iterator(logprob_batch_size):
                input_ids = lp_batch.get("input_ids")

                batch_size, seq_len = input_ids.shape

                # Create attention mask
                input_lengths = lp_batch.get("input_lengths")

                # Create attention mask for right-padded data
                attention_mask = torch.zeros(
                    (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                )
                for i, length in enumerate(input_lengths):
                    # For right-padded sequence, set 1s at the beginning of the sequence
                    attention_mask[i, :length] = 1

                # explicitly create position ids for the input, otherwise the sharding
                # for DTensor will be incorrect
                position_ids = torch.arange(seq_len, device=input_ids.device).repeat(
                    batch_size, 1
                )

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    # DTensor requires the casual attention kernel to hit,
                    # yet our attention mask above is not always all 1s
                    # this is fine because we mask with the actual attention mask
                    # later, but for input it has to be all 1s
                    attention_mask_input_all_ones = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_input_all_ones,
                        position_ids=position_ids,
                        use_cache=False,
                    )

                if isinstance(outputs.logits, DTensor):
                    token_logprobs = get_logprobs_from_vocab_parallel_logits(
                        outputs.logits.to(torch.float32), input_ids
                    )
                else:
                    # Extract logprobs for each token in the sequence by gathering the logprob
                    # corresponding to the next token at each position
                    # Input shapes:
                    #   log_probs: [batch_size, sequence_length, vocab_size] - logits for each position
                    #   token_ids: [batch_size, sequence_length] - actual tokens
                    # Output shape: [batch_size, sequence_length] - logprob of each token given previous
                    # We get logprob of token[t+1] from logits[t], prepending 0 to maintain sequence length

                    log_probs = torch.nn.functional.log_softmax(
                        outputs.logits.to(torch.float32), dim=-1
                    )
                    next_tokens = input_ids[:, 1:]
                    log_probs = log_probs[:, :-1]
                    token_logprobs = log_probs.gather(
                        dim=-1, index=next_tokens.unsqueeze(-1)
                    ).squeeze(-1)

                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )

                # Apply mask to zero out padding tokens logprobs
                token_logprobs = token_logprobs * attention_mask
                all_log_probs.append(token_logprobs)

        # Concatenate all batches
        return_data = BatchedDataDict()
        return_data["logprobs"] = torch.cat(all_log_probs, dim=0).cpu()

        return return_data

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        with torch.no_grad():
            try:
                curr_state_dict = get_cpu_state_dict(
                    self.model.state_dict().items(), pin_memory=True
                )
                curr_buffers = get_cpu_state_dict(
                    self.model.named_buffers(), pin_memory=True
                )

                for k, v in self.model.state_dict().items():
                    val = to_local_if_dtensor(v)
                    val.copy_(self.reference_model_state_dict[k])

                for k, v in self.model.named_buffers():
                    val = to_local_if_dtensor(v)
                    val.copy_(self.reference_model_buffers[k])

                yield

            finally:
                for k, v in self.model.state_dict().items():
                    val = to_local_if_dtensor(v)
                    val.copy_(curr_state_dict[k])

                for k, v in self.model.named_buffers():
                    val = to_local_if_dtensor(v)
                    val.copy_(curr_buffers[k])

    def get_reference_policy_logprobs(self, data: BatchedDataDict) -> BatchedDataDict:
        """Get the logprobs from the reference policy for a batch of data.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(data)

        return_data = BatchedDataDict()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

    def _add_noise_to_weights(self):
        """Add small Gaussian noise to the weights of the model. Note that this is used for testing purposes only."""
        noise_std = 0.01  # Standard deviation for the noise
        for p in self.model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)  # Add noise in-place
        torch.cuda.synchronize()

    def report_device_id(self) -> str:
        """Report the UUID of the current CUDA device using NVML.

        Returns:
            str: UUID of the device in the format "GPU-xxxxx"
        """
        from nemo_reinforcer.utils.nvml import get_device_uuid

        # Get current device index from torch
        device_idx = torch.cuda.current_device()
        # Get device UUID using NVML
        return get_device_uuid(device_idx)

    @torch.no_grad()
    def get_weight_ipc_handles(self, offload_model=True):
        from torch.multiprocessing.reductions import reduce_tensor

        self.model = self.move_to_cuda(self.model)
        params = self.model.state_dict()

        # Create a copy of parameters in the desired dtype (bfloat16 or float32)
        dtype_params = {}
        for name, param in params.items():
            if isinstance(param, DTensor):
                param = param.full_tensor()

            # Convert parameters to the configured dtype
            dtype_params[name] = param.to(
                device="cuda", dtype=self.dtype, non_blocking=True
            )

        for name, buffer in self.model.named_buffers():
            if isinstance(buffer, DTensor):
                buffer = buffer.full_tensor()

            dtype_params[name] = buffer.to(
                device="cuda", dtype=self.dtype, non_blocking=True
            )

        torch.cuda.synchronize()

        # Replace the original params with the converted ones
        params = dtype_params

        # hold on to the params so we can explicitly delete them after refit
        self._held_model_params = params

        data = {}
        device_uuid = self.report_device_id()
        for name, p in params.items():
            data[name] = reduce_tensor(p.detach())

        if offload_model or self.cpu_offload:
            self.model = self.move_to_cpu(self.model)
            gc.collect()
            torch.cuda.empty_cache()

        return {device_uuid: data}

    def prepare_for_lp_inference(self):
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.eval()
        self.offload_before_refit()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            # when cpu offload is enabled, the buffers do not get moved
            # to cuda automatically, so we need to do that manually
            self.model = self.move_buffer_to_device(self.model, "cuda")

        # have to move buffers to cuda manually for cpu offload case
        self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        # Move optimizer state to CUDA if it exists
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.cpu_offload
        ):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, (DTensor, torch.Tensor)):
                        state[k] = v.to("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_before_refit(self):
        """Offload the optimizer to the CPU."""
        torch.randn(1).cuda()  # wake up torch allocator
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, (DTensor, torch.Tensor)):
                        state[k] = v.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_after_refit(self):
        # Offload as much as possible on the CPU
        self.model = self.move_to_cpu(self.model)
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        if self._held_model_params is not None:
            del self._held_model_params
            self._held_model_params = None

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_to_device(self, model, device):
        model = self.move_buffer_to_device(model, device)
        return model.to(device)

    def move_buffer_to_device(self, model, device):
        # FSDP modules do not move buffers to the device automatically
        for v in model.buffers():
            v.data = v.data.to(device)

        return model

    def move_to_cuda(self, model):
        model = self.move_to_device(model, "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def move_to_cpu(self, model):
        model = self.move_to_device(model, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        save_torch_dist: bool = True,
        save_hf: bool = False,
    ):
        """Save a checkpoint of the model.

        the HuggingFace checkpoint is saved only if `save_hf` is True,
        and the optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        """
        save_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
            tokenizer=self.tokenizer if tokenizer_path else None,
            tokenizer_path=tokenizer_path,
            save_torch_dist=save_torch_dist,
            save_hf=save_hf,
        )

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a checkpoint into the model."""
        load_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
        )

    def shutdown(self):
        """Shutdown the policy."""
