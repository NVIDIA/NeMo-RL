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
import warnings
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import ray
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM

from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.algorithms.utils import get_tokenizer
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_reinforcer.models.generation.interfaces import (
    GenerationInterface,
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_reinforcer.models.interfaces import PolicyInterface
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.models.policy.utils import import_class_from_path
from nemo_reinforcer.distributed.virtual_cluster import (
    PY_EXECUTABLES,
)
from nemo_reinforcer.utils.native_checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


@ray.remote
class HfPolicyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.BASE

    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__name__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__name__}"

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
        tokenizer_name = self.cfg["tokenizer_name"]
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
        if init_reference_model:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",  # load weights onto CPU initially
                torch_dtype=torch.float32,  # use full precision in sft until https://github.com/NVIDIA/reinforcer/issues/13 is fixed
            )
        else:
            self.reference_model = None
        self.tokenizer = get_tokenizer(tokenizer_name)

        # ------------------------------------------------
        # 3) Move to GPU + Composable FSDP
        #    (Initialize device mesh, shard submodules, then shard entire model)
        # ------------------------------------------------

        def do_fsdp(model):
            # Create a device mesh with 'world_size' GPUs in a 1D arrangement.
            mesh = init_device_mesh("cuda", (world_size,))
            mp_policy = MixedPrecision(
                param_dtype=self.dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

            return FullyShardedDataParallel(
                model,
                device_mesh=mesh,
                auto_wrap_policy=size_based_auto_wrap_policy,
                mixed_precision=mp_policy,
            )

        self.model.to("cuda")
        self.model = do_fsdp(self.model)
        self.model = self.move_to_cpu(self.model)
        if self.reference_model is not None:
            self.reference_model.to("cuda")
            self.reference_model = do_fsdp(self.reference_model)
            self.reference_model = self.move_to_cpu(self.reference_model)
        self.model.to("cuda")
        self._held_reference_model_params = None
        # register_fsdp_forward_method(self.model, "generate")
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
            self.load_checkpoint(
                weights_path,
                optimizer_path,
            )
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
        local_gbs = gbs // torch.distributed.get_world_size()
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
                input_ids = mb.get("input_ids")

                input_lengths = mb.get("input_lengths")
                batch_size, seq_len = input_ids.shape
                attention_mask = torch.ones(
                    (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                )
                for i, length in enumerate(input_lengths):
                    # For right-padded sequence, set 1s at the beginning of the sequence
                    attention_mask[i, :length] = 1

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
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

                # Loss is accumulated across microbatches, so we need to scale by the number of microbatches
                loss = loss / num_microbatches
                if not eval_mode:
                    loss.backward()
                mb_losses.append(loss.item())
                all_mb_metrics.append(loss_metrics)

            # Clip gradients
            if not eval_mode:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update parameters
                self.optimizer.step()
                self.scheduler.step()
            losses.append(torch.tensor(mb_losses).sum().item())

        # Compute global loss across all ranks
        with torch.no_grad():
            local_loss = torch.tensor(losses, device="cuda")
            global_loss = torch.zeros_like(local_loss)
            torch.distributed.all_reduce(local_loss)
            global_loss = local_loss / torch.distributed.get_world_size()

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

        # Process in batches
        with torch.no_grad():
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
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                log_probs = torch.nn.functional.log_softmax(
                    outputs.logits.to(torch.float32), dim=-1
                )

                # Extract logprobs for each token in the sequence by gathering the logprob
                # corresponding to the next token at each position
                # Input shapes:
                #   log_probs: [batch_size, sequence_length, vocab_size] - logits for each position
                #   token_ids: [batch_size, sequence_length] - actual tokens
                # Output shape: [batch_size, sequence_length] - logprob of each token given previous
                # We get logprob of token[t+1] from logits[t], prepending 0 to maintain sequence length
                token_ids = input_ids
                next_tokens = token_ids[:, 1:]  # Skip first token
                log_probs = log_probs[:, :-1]  # Remove last position's logits
                token_logprobs = log_probs.gather(
                    dim=-1, index=next_tokens.unsqueeze(-1)
                ).squeeze(-1)

                # Prepend 0 logprob for first token to maintain same sequence length as input
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
        try:
            # Save original references
            original_model = self.model
            original_reference_model = self.reference_model

            self.model = self.move_to_cpu(self.model)
            self.reference_model = self.reference_model.to("cuda")

            # Swap the references
            self.model, self.reference_model = self.reference_model, self.model
            gc.collect()
            torch.cuda.empty_cache()

            # - self.model is the original reference_model, now on CUDA
            # - self.reference_model is the original model, now on CPU
            yield

        finally:
            # Restore original references and device placement
            self.reference_model = self.move_to_cpu(original_reference_model)
            self.model = original_model.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()

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

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to vLLM worker is not properly right-padded: {error_msg}"
            )

        self.model.eval()

        # Right padded tokens are converted to left padded tokens for HF generate (https://huggingface.co/docs/transformers/main/en/llm_tutorial?padding=right+pad#padding-side)
        with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(
            self.model, recurse=False
        ):
            # Get generation config from self.cfg
            generation_batch_size = self.cfg["generation_batch_size"]
            gen_cfg = self.cfg["generation"]
            stop_strings = []
            if data["stop_strings"] is not None:
                stop_strings += [
                    string for strings in data["stop_strings"] for string in strings
                ]
            if gen_cfg["stop_strings"] is not None:
                stop_strings += gen_cfg["stop_strings"]

            micro_batches = []

            # Process in batches
            max_length = 0
            for gen_batch in data.make_microbatch_iterator(generation_batch_size):
                # Create attention mask from input_lengths if needed for the model
                input_ids = gen_batch.get("input_ids").cuda()
                input_lengths = gen_batch.get("input_lengths").cuda()
                batch_size, seq_len = input_ids.shape

                # Convert right padding to left padding
                left_padded_input_ids = torch.full_like(
                    input_ids, gen_cfg["pad_token_id"]
                )
                left_padded_attention_mask = torch.zeros(
                    (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                )

                for i, length in enumerate(input_lengths):
                    # Move tokens to the end of the sequence (left padding)
                    left_padded_input_ids[i, seq_len - length :] = input_ids[i, :length]
                    # Set attention mask for the actual tokens (at the end for left padding)
                    left_padded_attention_mask[i, seq_len - length :] = 1

                outputs = self.model.module.generate(
                    input_ids=left_padded_input_ids,
                    attention_mask=left_padded_attention_mask,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    do_sample=not greedy,
                    temperature=gen_cfg["temperature"],
                    top_p=gen_cfg["top_p"],
                    top_k=gen_cfg["top_k"],
                    pad_token_id=gen_cfg["pad_token_id"],
                    eos_token_id=gen_cfg["stop_token_ids"],
                    stop_strings=stop_strings,
                    tokenizer=self.tokenizer,  # needs for stop_strings
                    return_dict_in_generate=True,
                    output_scores=True,
                    synced_gpus=True,
                )
                # Get the generated sequences
                max_length = max(max_length, outputs.sequences.size(1))

                # Convert scores to log probabilities and extract the logprob of the chosen token
                scores = torch.stack(
                    outputs.scores, dim=1
                )  # [batch_size, seq_len, vocab_size]
                logprobs = torch.nn.functional.log_softmax(scores, dim=-1)

                # Get the logprobs of the actually generated tokens
                # outputs.sequences[:, -scores.size(1):] gives us just the newly generated tokens
                generated_tokens = outputs.sequences[:, -scores.size(1) :]
                token_logprobs = logprobs.gather(
                    dim=-1, index=generated_tokens.unsqueeze(-1)
                ).squeeze(-1)

                # Prepend zeros for input tokens based on original input lengths, not the padded length
                mb = {}
                mb["orig_input_lengths"] = input_lengths.clone()
                mb["generation_logprobs"] = token_logprobs
                mb["left_padded_output_ids"] = outputs.sequences

                micro_batches.append(mb)

            # Get lengths, pad, and concatenate all batches
            return_data = BatchedDataDict.from_batches(
                micro_batches,
                pad_value_dict={
                    "left_padded_output_ids": self.cfg["generation"]["pad_token_id"]
                },
            )

            # Calculate the lengths of generations for each sequence by finding stop tokens
            generation_lengths = []
            unpadded_sequence_lengths = []
            input_length = data.get("input_ids").size(1)

            # Convert left-padded outputs back to right-padded format
            batch_size = len(return_data["left_padded_output_ids"])
            max_seq_len = max(
                [seq.size(0) for seq in return_data["left_padded_output_ids"]]
            )
            right_padded_output_ids = torch.full(
                (batch_size, max_seq_len),
                self.cfg["generation"]["pad_token_id"],
                dtype=return_data["left_padded_output_ids"][0].dtype,
                device=return_data["left_padded_output_ids"][0].device,
            )

            for idx, seq in enumerate(return_data["left_padded_output_ids"]):
                # Get only the generated part (excluding input)
                original_length = return_data["orig_input_lengths"][idx].item()
                seq_len = seq.size(0)

                # The generated content starts after the left-padded input
                generated_part = seq[-(seq_len - input_length) :]

                eos_positions = (generated_part == self.tokenizer.eos_token_id).nonzero(
                    as_tuple=True
                )[0]
                # TODO @sahilj: handle different stopping criteria
                # Calculate generation length
                if len(eos_positions) > 0:
                    gen_length = (
                        eos_positions[0].item() + 1
                    )  # +1 to include the EOS token
                else:
                    gen_length = len(generated_part)

                generation_lengths.append(gen_length)

                valid_length = original_length + gen_length
                unpadded_sequence_lengths.append(valid_length)

                # Extract the original input tokens from the left-padded sequence
                # For left-padded sequences, tokens are at the end of the input section
                valid_input_part = (
                    seq[input_length - original_length : input_length]
                    if original_length > 0
                    else torch.tensor([], device=seq.device, dtype=seq.dtype)
                )

                # Combine with generated part
                valid_generated_part = generated_part[:gen_length]
                valid_tokens = torch.cat([valid_input_part, valid_generated_part])

                # Place at the beginning of the right-padded sequence
                right_padded_output_ids[idx, :valid_length] = valid_tokens

            # Store the right-padded outputs
            return_data["output_ids"] = right_padded_output_ids

            # Align generation_logprobs with right-padded output format
            batch_size = len(return_data["generation_logprobs"])
            right_padded_logprobs = torch.zeros(
                (batch_size, max_seq_len),
                dtype=return_data["generation_logprobs"][0].dtype,
                device=return_data["generation_logprobs"][0].device,
            )

            for idx, logprob_seq in enumerate(return_data["generation_logprobs"]):
                original_length = return_data["orig_input_lengths"][idx].item()
                gen_length = generation_lengths[idx]

                # For right-padded format, we need:
                # 1. Zeros for the original input tokens (at the beginning)
                # 2. Actual logprobs for generated tokens (after the zeros)
                # 3. Zeros padding at the end (if needed)

                right_padded_seq = torch.zeros(
                    max_seq_len, dtype=logprob_seq.dtype, device=logprob_seq.device
                )
                right_padded_seq[original_length : original_length + gen_length] = (
                    logprob_seq[:gen_length]
                )
                right_padded_logprobs[idx] = right_padded_seq
                valid_length = original_length + gen_length

            # Remove the temporary data we added
            if "generation_logprobs" in return_data:
                del return_data["generation_logprobs"]
            if "orig_input_lengths" in return_data:
                del return_data["orig_input_lengths"]
            if "left_padded_output_ids" in return_data:
                del return_data["left_padded_output_ids"]

            # Ensure consistent data types and device placement
            return_data["output_ids"] = right_padded_output_ids
            return_data["logprobs"] = right_padded_logprobs
            return_data["generation_lengths"] = torch.tensor(
                generation_lengths, dtype=torch.long
            )
            return_data["unpadded_sequence_lengths"] = torch.tensor(
                unpadded_sequence_lengths, dtype=torch.long
            )

            # Move everything to CPU before returning
            return_data.to("cpu")

            return return_data

    def _add_noise_to_weights(self):
        """Add small Gaussian noise to the weights of the model. Note that this is used for testing purposes only."""
        # TODO @sahilj: do this without a summon (maybe FSDP2)
        noise_std = 0.01  # Standard deviation for the noise
        with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(
            self.model, recurse=True
        ):
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

        # TODO @sahilj: do this without an allgather (maybe FSDP2)
        params = self.model.state_dict()

        # Create a copy of parameters in the desired dtype (bfloat16 or float32)
        dtype_params = {}
        for name, param in params.items():
            # Convert parameters to the configured dtype
            dtype_params[name] = param.to(self.dtype, non_blocking=True)

        # Replace the original params with the converted ones
        params = dtype_params
        self._held_reference_model_params = params
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
        self.model.to("cuda")
        self.model.eval()
        self.offload_before_refit()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model.to("cuda")
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

        if self._held_reference_model_params is not None:
            del self._held_reference_model_params
            self._held_reference_model_params = None

        gc.collect()
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_to_cpu(self, model):
        for param in model.parameters():
            param.data = param.data.to("cpu")

        for buffer in model.buffers():
            buffer.data = buffer.data.to("cpu")

        if hasattr(model, "_fsdp_wrapped_module"):
            model._fsdp_wrapped_module.to("cpu")

        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        save_torch_dist: bool = True,
        save_hf: bool = False,
    ):
        """Save a checkpoint of the model.

        The checkpoint is saved in the following format:

        weights_path/
            __0_1.distcp
            __1_0.distcp
            ...
        weights_path-hf/
            config.json
            generation_config.json
            model-00001-of-<TOTAL_SHARDS>.safetensors
            ...
            model.safetensors.index.json
        optimizer_path/
            __0_0.distcp
            __1_0.distcp
            ...

        the HuggingFace checkpoint is saved only if `save_hf` is True,
        and the optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        """
        save_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
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
        #
        pass


class HfPolicy(PolicyInterface, GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        name_prefix: str = "hf_policy",
        workers_per_node: Optional[Union[int, List[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_reference_model: bool = True,
    ):
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        worker_builder = RayWorkerBuilder(
            HfPolicyWorker,
            config,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_reference_model=init_reference_model,
        )
        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            workers_per_node=workers_per_node,
        )
        self.dp_size = self.worker_group.world_size
        self.cfg = config

    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_multiple_data(
            "get_logprobs", sharded_data
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )
        return logprobs

    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_multiple_data(
            "get_reference_policy_logprobs", sharded_data
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )
        return logprobs

    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ):
        """Train the policy on a batch of data with a given loss function."""
        # Shard and replicate the batch
        shards = self.dp_size
        sharded_data = data.shard_by_batch_size(
            shards, batch_size=self.cfg["train_global_batch_size"]
        )

        # Train each shard in parallel
        futures = self.worker_group.run_all_workers_multiple_data(
            "train",
            sharded_data,
            common_kwargs={
                "loss_fn": loss_fn,
                "eval_mode": eval_mode,
                "gbs": gbs,
                "mbs": mbs,
            },
        )
        results = self.worker_group.get_all_worker_results(futures)

        # Aggregate the results
        aggregated_results = {}
        aggregated_results["loss"] = results[0]["global_loss"]

        # Aggregate metrics across all workers
        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r["all_mb_metrics"].items():
                all_mb_metrics[k].extend(v)
        aggregated_results["all_mb_metrics"] = dict(all_mb_metrics)

        return aggregated_results

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the policy."""
        # Verify input data is right-padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "Missing required input fields"
        )

        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_multiple_data(
            "generate", sharded_data, common_kwargs={"greedy": greedy}
        )
        result = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
            pad_value_dict={"output_ids": self.cfg["generation"]["pad_token_id"]},
        )

        # Verify the output has all required fields
        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return result

    def prepare_for_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def prepare_for_training(self, *args, **kwargs):
        # onload everything to the GPU
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_training", respect_tied_workers=True
        )
        ray.get(futures)
        pass

    def prepare_for_lp_inference(self, *args, **kwargs):
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference", respect_tied_workers=True
        )
        ray.get(futures)

    def finish_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def finish_training(self, *args, **kwargs):
        # Placeholder implementation
        pass

    def get_weights_ipc_handles(self):
        """Fetch weight IPC handles from all workers.

        Returns:
            dict: A dictionary mapping device UUIDs to parameter IPC handles.
        """
        # Collect IPC handles from all workers
        worker_handles = ray.get(
            [
                worker.get_weight_ipc_handles.remote()
                for worker in self.worker_group.workers
            ]
        )

        # Combine all worker handles into a single dictionary
        all_handles = {}
        for handle in worker_handles:
            all_handles.update(handle)

        return all_handles

    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_before_refit", respect_tied_workers=True
        )
        ray.get(futures)

    def offload_after_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_after_refit", respect_tied_workers=True
        )
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        save_torch_dist: bool = True,
        save_hf: bool = False,
    ):
        """Save a checkpoint of the model."""
        futures = self.worker_group.run_all_workers_single_data(
            "save_checkpoint",
            weights_path,
            optimizer_path,
            save_torch_dist,
            save_hf,
            respect_tied_workers=True,
        )
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shut down all HF workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def __del__(self):
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call worker_group.shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls worker_group.shutdown().
        """
        self.worker_group.shutdown()
