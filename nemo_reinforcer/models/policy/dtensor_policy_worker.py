import gc

import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import ray
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FSDPModule,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemo_reinforcer.models.dtensor.parallelize import _parallelize_model

from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.policy.utils import import_class_from_path
from nemo_reinforcer.distributed.virtual_cluster import (
    PY_EXECUTABLES,
)

from torch.distributed.tensor import DTensor
from nemo_reinforcer.models.dtensor.parallelize import (
    get_logprobs_from_vocab_parallel_logits,
)


@contextmanager
def unshard_fsdp2_model(model):
    try:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()


@ray.remote
class DTensorPolicyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.DEFAULT_VENV

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
        self.cpu_offload = self.cfg["cpu_offload"]

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # If no pad token is defined, you might need:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------------------------------------------------
        # 3) Move to GPU + Composable FSDP
        #    (Initialize device mesh, shard submodules, then shard entire model)
        # ------------------------------------------------

        tp_size = self.cfg["tensor_parallel_size"]
        assert tp_size > 1, "TP size must be greater than 1 to use DTensor"
        dp_size = world_size // tp_size
        assert world_size % tp_size == 0, (
            "World size must be divisible by TP size to use DTensor"
        )

        mesh_2d = init_device_mesh(
            "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh, tp_mesh = mesh_2d["dp"], mesh_2d["tp"]

        self.dp_size = dp_size
        self.tp_size = tp_size
        self.dp_mesh = dp_mesh
        self.tp_mesh = tp_mesh

        print("### BEFORE MOVE TO CUDA")
        self.model = self.move_to_cuda(self.model)
        print("### AFTER MOVE TO CUDA")
        self.model = _parallelize_model(
            self.model,
            self.dp_mesh,
            self.tp_mesh,
            param_dtype=self.dtype,
            sequence_parallel=self.cfg["sequence_parallel"],
            cpu_offload=self.cpu_offload,
            activation_checkpointing=self.cfg["activation_checkpointing"],
        )
        self.model = self.move_to_cpu(self.model)
        self._held_model_params = None

        if init_reference_model:
            # self._held_reference_model_params = self.model.state_dict()
            self._held_reference_model_params = None

        # register_fsdp_forward_method(self.model, "generate")
        if init_optimizer:
            optimizer_cls = import_class_from_path(self.cfg["optimizer"]["name"])
            self.optimizer = optimizer_cls(
                self.model.parameters(), **self.cfg["optimizer"]["kwargs"]
            )
        else:
            self.optimizer = None

        if "scheduler" in self.cfg:
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

        else:
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
        for module_name, module in self.model.named_modules():
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
            "parameter_sample": param_info,
            "env_vars": {
                k: v
                for k, v in os.environ.items()
                if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
            },
        }

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
            print("### LOCAL GBS", local_gbs, gb_start, dataset_size)
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
                    # input_ids = torch.load("/lustre/fsw/portfolios/llmservice/users/geshen/newer_reinforcer/reinforcer/0_ratios_error.pt", map_location="cuda")["input_ids"]
                    batch_size, seq_len = input_ids.shape

                    attention_mask_input = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )
                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_input,
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

            if not eval_mode:
                # Clip gradients
                if (
                    not self.cpu_offload
                ):  # cpu offload doesn't support grad norm clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
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
            global_loss = local_loss / self.dp_size

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        metrics = {
            "global_loss": global_loss.cpu(),
            "local_loss": local_loss.cpu(),
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

        context_mgr = unshard_fsdp2_model(self.model)
        # Process in batches
        with context_mgr, torch.no_grad():
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

                # Process with the model directly using right-padded inputs
                position_ids = torch.arange(seq_len, device=input_ids.device).repeat(
                    batch_size, 1
                )

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    attention_mask_input = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )
                    print("### INPUT IDS", input_ids.sum(), input_ids.shape)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_input,
                        position_ids=position_ids,
                        use_cache=False,
                    )

                token_logprobs = get_logprobs_from_vocab_parallel_logits(
                    outputs.logits, input_ids
                )
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
        # TODO
        try:
            yield

        finally:
            ...

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

    def zero_out_weights(self):
        """Zero out the weights of the model."""
        for v in self.model.parameters():
            v.zero_()

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

        # TODO @sahilj: do this without an allgather (maybe FSDP2)
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

        # Replace the original params with the converted ones
        params = dtype_params

        # hold on to the params so we can explicitly delete them after refit
        self._held_model_params = params

        data = {}
        device_uuid = self.report_device_id()
        for name, p in params.items():
            data[name] = reduce_tensor(p.detach())

        if offload_model:
            self.model = self.move_to_cpu(self.model)
            gc.collect()
            torch.cuda.empty_cache()
        return {device_uuid: data}

    def prepare_for_lp_inference(self):
        self.move_to_cuda(self.model)
        self.model.eval()
        self.offload_before_refit()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.move_to_cuda(self.model)
        self.model.train()

        # Move optimizer state to CUDA if it exists
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v) and not v.is_cuda:
                        state[k] = v.to("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        torch.randn(1).cuda()  # wake up torch allocator
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to("cpu")

        for buffer in self.model.buffers():
            buffer.data = buffer.data.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

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

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_to_device(self, model, device):
        return model.to(device)

    def move_to_cuda(self, model):
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before moving to cuda: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        model = self.move_to_device(model, "cuda")

        gc.collect()
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after moving to cuda: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        return model

    def move_to_cpu(self, model):
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before moving to cpu: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        model = self.move_to_device(model, "cpu")
        gc.collect()
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after moving to cpu: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        offload_to_cpu: bool = True,
    ):
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        scheduler_state_dict = self.scheduler.state_dict()

        optim_and_scheduler_state_dict = {
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
        }

        if torch.distributed.get_rank() == 0:
            # check if weights_path dir exists
            weights_dir = os.path.dirname(weights_path)
            if not os.path.exists(weights_dir):
                print(f"Creating weights directory {weights_dir} DOESN'T EXIST SOMEHOW")
                os.makedirs(weights_dir)

        torch.distributed.barrier()
        torch.distributed.checkpoint.save(model_state_dict, checkpoint_id=weights_path)

        if optimizer_path is not None:
            torch.distributed.checkpoint.save(
                optim_and_scheduler_state_dict, checkpoint_id=optimizer_path
            )

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        print(f"Loading Policy from {weights_path} and optimizer from {optimizer_path}")
        optimizer_state_dict = None

        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        scheduler_state_dict = self.scheduler.state_dict()

        torch.distributed.checkpoint.load(model_state_dict, checkpoint_id=weights_path)

        if optimizer_path is not None:
            torch.distributed.checkpoint.load(
                {"optimizer": optimizer_state_dict, "scheduler": scheduler_state_dict},
                checkpoint_id=optimizer_path,
            )
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=model_state_dict,
                optim_state_dict=optimizer_state_dict,
            )

            if scheduler_state_dict is not None:
                self.scheduler.load_state_dict(scheduler_state_dict)
        else:
            # technically because the state dict is mutated we don't need this line,
            # but we're paranoid
            self.model.load_state_dict(model_state_dict)

    def shutdown(self):
        """Shutdown the policy."""
        #
        pass
