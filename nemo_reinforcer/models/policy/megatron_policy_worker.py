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
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Iterable
from functools import partial

import ray
from ray.util.queue import Queue
import torch
import time

from transformers import AutoTokenizer

from nemo.tron.init import initialize_megatron
from nemo.tron.config import (
    ConfigContainer,
    TrainingConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    CheckpointConfig,
    DistributedDataParallelConfig,
)
from nemo.tron.utils.common_utils import get_rank_safe
from nemo.tron.config import TokenizerConfig
from nemo.tron.model import get_model_from_config
from nemo.tron.checkpointing import checkpoint_exists, load_checkpoint
from nemo.tron.init import initialize_megatron, set_jit_fusion_options
from nemo.tron.setup import _init_checkpointing_context, _update_model_config_funcs
from nemo.tron.state import GlobalState
from nemo.tron.optim import setup_optimizer
from nemo.tron import fault_tolerance
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.train_utils import (
    calc_params_l2_norm,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)


from megatron.core import parallel_state
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core.models.gpt import GPTModel
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from nemo_reinforcer.models.megatron.common import forward_step_arbitrary_loss
from nemo.tron.train import train_step
from nemo.tron.setup import HAVE_FSDP2

from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.inference.text_generation.mcore_engine_server import (
    ModelInferenceWrapperServer,
    run_mcore_engine,
)
from megatron.core.inference.engines import (
    StaticInferenceEngine,
)
from megatron.core.inference.sampling_params import SamplingParams
from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict
from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_reinforcer.models.generation.interfaces import (
    GenerationInterface,
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_reinforcer.models.policy import PolicyConfig
from nemo_reinforcer.distributed.virtual_cluster import (
    PY_EXECUTABLES,
)
from nemo_reinforcer.distributed.named_sharding import NamedSharding

from nemo_reinforcer.models.policy.utils import get_gpu_info
from nemo_reinforcer.distributed.model_utils import from_parallel_logits_to_logprobs


def setup_megatron_model(
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        gpu_visibility_externally_set=True,
    )

    if cfg.ft_config and cfg.ft_config.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft_config)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model_config, cfg.train_config.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = _init_checkpointing_context(cfg.checkpoint_config)

    # Tokenizer
    tokenizer = build_tokenizer(
        cfg.tokenizer_config,
        make_vocab_size_divisible_by=cfg.model_config.make_vocab_size_divisible_by,
        tensor_model_parallel_size=cfg.model_config.tensor_model_parallel_size,
    )
    if not cfg.model_config.vocab_size:
        cfg.model_config.vocab_size = tokenizer.vocab_size

    torch.distributed.barrier()

    # Model, optimizer, and learning rate.
    model = get_model_from_config(
        cfg.model_config,
        cfg.ddp_config,
        use_torch_fsdp2=cfg.dist_config.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer_config.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng_config.data_parallel_random_init,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer_config,
            scheduler_config=cfg.scheduler_config,
            model=model,
            use_gloo_process_groups=cfg.dist_config.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    _update_model_config_funcs(
        model,
        cfg.model_config,
        cfg.ddp_config,
        optimizer,
        align_grad_reduce=cfg.dist_config.align_grad_reduce,
    )
    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint_config.load is not None
        or cfg.checkpoint_config.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint_config.load)
        or checkpoint_exists(cfg.checkpoint_config.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist_config.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


@ray.remote
class MegatronPolicyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

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
        checkpoint_dir: str,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        load_reference_model: bool = True,
    ):
        self.cfg = config
        self.checkpoint_dir = checkpoint_dir

        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")

        hf_model_name = self.cfg["model_name"]
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if the checkpoint already exists
        output_path = f"/opt/checkpoints/tron/{hf_model_name}"
        pt_checkpoint_exists = os.path.exists(output_path) and os.path.exists(
            os.path.join(output_path, "iter_0000000")
        )

        if get_rank_safe() == 0:
            if pt_checkpoint_exists:
                print(f"Checkpoint already exists at {output_path}. Skipping import.")
            else:
                if "llama" in hf_model_name.lower():
                    from nemo.tron.converter.llama import HFLlamaImporter

                    print(f"Importing model {hf_model_name} to {output_path}...")
                    importer = HFLlamaImporter(
                        hf_model_name,
                        output_path=f"/opt/checkpoints/tron/{hf_model_name}",
                    )
                elif "qwen" in hf_model_name.lower():
                    from nemo.tron.converter.qwen import HFQwen2Importer

                    print(f"Importing model {hf_model_name} to {output_path}...")
                    importer = HFQwen2Importer(
                        hf_model_name,
                        output_path=f"/opt/checkpoints/tron/{hf_model_name}",
                    )
                else:
                    raise ValueError(f"Unknown model: {hf_model_name}")
                importer.apply()
                import megatron.core.rerun_state_machine

                megatron.core.rerun_state_machine.destroy_rerun_state_machine()
            pre_init_communication_queue.put(True)
        else:
            pre_init_communication_queue.get()
            pre_init_communication_queue.put(True)

        pretrained_ckpt = f"/opt/checkpoints/tron/{hf_model_name}"
        pretrained_run_config = os.path.join(
            pretrained_ckpt, "iter_0000000/run_config.yaml"
        )
        cfg_from_pretrained = ConfigContainer.from_yaml(pretrained_run_config)
        model_cfg = cfg_from_pretrained.model_config
        cfg_from_pretrained.logger_config = LoggerConfig()
        cfg_from_pretrained.checkpoint_config = CheckpointConfig(
            save_interval=100,
            save="/nemo_run/checkpoints",
            load="/nemo_run/checkpoints",
            pretrained_checkpoint=pretrained_ckpt,  # This is the path to the pretrained ckpt for the SFT case
            async_save=True,
            fully_parallel_save=True,
        )

        model_cfg.tensor_model_parallel_size = self.cfg["tensor_model_parallel_size"]
        model_cfg.pipeline_model_parallel_size = self.cfg[
            "pipeline_model_parallel_size"
        ]
        model_cfg.context_parallel_size = self.cfg[
            "context_parallel_size"
        ]  # not supported right now
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        model_cfg.params_dtype = torch.bfloat16  # amp
        model_cfg.parallel_output = True

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save="/nemo_run/checkpoints",
            load="/nemo_run/checkpoints",
            pretrained_checkpoint=pretrained_ckpt,  # This is the path to the pretrained ckpt for the SFT case
            async_save=True,
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_ckpt,  # This is the path to the pretrained ckpt for the SFT case
            fully_parallel_load=True,  # Enable fully parallel load
        )
        self.megatron_cfg = ConfigContainer(
            model_config=model_cfg,
            checkpoint_config=checkpoint_config,
            logger_config=LoggerConfig(logging_level=0),
            train_config=TrainingConfig(
                micro_batch_size=self.cfg["train_micro_batch_size"],  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=1000,  # Default value for inference
            ),
            optimizer_config=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp_config=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=True,
                use_distributed_optimizer=True,
            ),
            scheduler_config=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset_config=None,
            tokenizer_config=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        self.megatron_cfg.validate()

        print(f"cfg: {self.megatron_cfg}")
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(self.megatron_cfg, load_optimizer=True)
        self.model = self.model[0]  # Get the first model from the list
        for name, item in self.model.state_dict().items():
            if isinstance(item, torch.Tensor):
                item = item.detach().to(device="cpu", non_blocking=True, copy=True)
            self.model.state_dict()[name] = item

        if load_reference_model:
            ref_ckpt_context = _init_checkpointing_context(ref_checkpoint_config)
            reference_model = get_model_from_config(
                self.megatron_cfg.model_config,
                self.megatron_cfg.ddp_config,
                use_torch_fsdp2=self.megatron_cfg.dist_config.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer_config.overlap_param_gather_with_optimizer_step,
                data_parallel_random_init=self.megatron_cfg.rng_config.data_parallel_random_init,
            )
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    self.mcore_state,
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist_config.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    self.reference_state_dict[name] = item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

        for name, item in self.model.state_dict().items():
            if isinstance(item, torch.Tensor):
                item = item.detach().to(device="cuda", non_blocking=True, copy=True)
            self.model.state_dict()[name] = item

        from nemo.tron.tokenizers.tokenizer import build_tokenizer

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=self.cfg["tensor_model_parallel_size"],
        )
        self.final_padded_vocab_size = tokenizer_config.padded_vocab_size
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")

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
        dataset_size = data.size

        if eval_mode:
            ctx = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            forward_step = partial(forward_step_arbitrary_loss, loss_fn=loss_fn)
            all_mb_metrics = []
            for gb_start in range(0, dataset_size, local_gbs):
                num_microbatches = local_gbs // mbs
                data_iterator = data.slice(
                    gb_start, gb_start + local_gbs
                ).make_microbatch_iterator(mbs)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(forward_step, self.mcore_state),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=num_microbatches,
                        seq_length=self.cfg[
                            "max_total_sequence_length"
                        ],  # model_config.seq_length,
                        micro_batch_size=self.cfg["train_micro_batch_size"],
                        decoder_seq_length=self.cfg[
                            "max_total_sequence_length"
                        ],  # model_config.seq_length,
                        forward_only=False,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
                num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                # Update learning rate.
                if update_successful:
                    increment = (
                        num_microbatches
                        * self.cfg["train_micro_batch_size"]
                        * self.dp_size
                    )
                    self.scheduler.step(increment=increment)
                    skipped_iter = 0
                    curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                    curr_wd = self.scheduler.get_wd()
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # Average loss across microbatches.
                    loss_metrics = {}
                    for key in losses_reduced[0].keys():
                        numerator = 0
                        denominator = 0
                        for x in losses_reduced:
                            val = x[key]
                            # there is one dict per microbatch. in new reporting, we average
                            # over the total number of tokens across the global batch.
                            if isinstance(val, tuple) or isinstance(val, list):
                                numerator += val[0]
                                denominator += val[1]
                            else:
                                # legacy behavior. we average over the number of microbatches,
                                # and so the denominator is 1.
                                numerator += val
                                denominator += 1
                        loss_metrics[key] = numerator / denominator

                loss_metrics["lr"] = curr_lr
                loss_metrics["wd"] = curr_wd

                print(f"Skipped iterations: {skipped_iter}")
                all_mb_metrics.append(loss_metrics)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            loss = torch.tensor(loss_metrics["loss"], device="cuda")
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        metrics = {
            "global_loss": loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "all_mb_metrics": dict(mb_metrics),
        }
        return metrics

    def get_logprobs(
        self, data: BatchedDataDict, micro_batch_size: int = None
    ) -> BatchedDataDict:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.

        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.

        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        all_log_probs = []
        self.model.eval()

        with torch.no_grad():

            def forward_step_fn(data_iterator: Iterable, model: GPTModel):
                data_dict = next(data_iterator)
                input_ids = data_dict["input_ids"].cuda()
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    input_ids, 0, False, False, False
                )
                output_tensor = model(input_ids, position_ids, attention_mask)

                def collection_fn(output_tensor):
                    tp_grp = get_tensor_model_parallel_group()
                    tp_rank = get_tensor_model_parallel_rank()
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor.to(torch.float32),
                        target=input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                    )

                    # Prepend 0 logprob for first token to maintain same sequence length as input
                    token_logprobs = torch.cat(
                        [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                    )

                    return torch.tensor(0.0), {"logprobs": token_logprobs}

                return output_tensor, collection_fn

            forward_backward_func = get_forward_backward_func()
            list_of_logprobs = forward_backward_func(
                forward_step_func=forward_step_fn,
                data_iterator=data.make_microbatch_iterator(logprob_batch_size),
                model=self.model,
                num_microbatches=max(1, data.size // logprob_batch_size),
                seq_length=self.cfg["max_total_sequence_length"],
                micro_batch_size=logprob_batch_size,
                decoder_seq_length=self.cfg["max_total_sequence_length"],
                forward_only=True,
            )
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            logprobs = torch.cat(all_logprobs, dim=0)
            return BatchedDataDict(logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                gc.collect()
                torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                gc.collect()
                torch.cuda.empty_cache()

    def get_reference_policy_logprobs(
        self, data: BatchedDataDict, micro_batch_size: int = None
    ) -> BatchedDataDict:
        """Get the logprobs from the reference policy for a batch of data.

        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(data, micro_batch_size)

        return_data = BatchedDataDict()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # self.model.config.flash_decode = True
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
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model_config
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=False,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        inference_wrapped_model = ModelInferenceWrapperServer(
            self.model, inference_wrapper_config
        )
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )
        inference_engine = StaticInferenceEngine(
            text_generation_controller=text_generation_controller,
            max_batch_size=self.cfg["generation_batch_size"],
        )

        # apply chat template
        out = run_mcore_engine(
            engine=inference_engine,
            prompt_tokens_tensor=data.get("input_ids"),
            prompt_lengths_tensor=data.get("input_lengths"),
            tokens_to_generate=self.cfg["generation"]["max_new_tokens"]
            - data.get("input_ids").size(1),
        )
        # print(out)

        input_lengths = data.get("input_lengths")
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data.get("input_ids").size(0)
        max_seq_len = max([len(tokens) for tokens in out["tokens"]])

        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data.get("input_ids").device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data.get("input_ids").device,
        )

        # Fill in the padded tensors with actual values
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data.get("input_ids").device
            )

            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data.get("input_ids").device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": torch.tensor(
                [len(o) - input_lengths[i] for i, o in enumerate(out["logprobs"])]
            ),
            "unpadded_sequence_lengths": torch.tensor(
                [len(o) for o in out["logprobs"]]
            ),
        }

        self.model.config.flash_decode = False
        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    def zero_out_weights(self):
        """Zero out the weights of the model."""
        pass

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
        pass

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
            # for state in self.optimizer.state.values():
            for state in self.optimizer._get_state().values():
                for k, v in state.items():
                    if torch.is_tensor(v) and not v.is_cuda:
                        state[k] = v.to("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        return
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
        return
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
        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        offload_to_cpu: bool = True,
    ):
        pass

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        pass

    def shutdown(self):
        """Shutdown the policy."""
        #
        pass
