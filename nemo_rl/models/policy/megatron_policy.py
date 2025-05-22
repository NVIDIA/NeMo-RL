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
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import ray
from ray.util.queue import Queue
from transformers import AutoTokenizer
import torch

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.interfaces import PolicyInterface
from nemo_rl.models.policy import PolicyConfig


class MegatronPolicy(PolicyInterface, GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        tokenizer: AutoTokenizer,
        name_prefix: str = "megatron_policy",
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

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                (
                    config["pipeline_model_parallel_size"],
                    -1,
                    config["tensor_model_parallel_size"],
                )
            ),
            names=["pipeline_model_parallel", "data_parallel", "tensor_model_parallel"],
        )

        pre_init_queue = (
            Queue()
        )  # just for communication before torch distributed is set up
        worker_builder = RayWorkerBuilder(
            "nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker",
            config,
            tokenizer=tokenizer,
            checkpoint_dir=weights_path,
            worker_sharding_annotations=self.sharding_annotations,
            pre_init_communication_queue=pre_init_queue,
            init_optimizer=init_optimizer,
            init_reference_model=init_reference_model,
        )

        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            workers_per_node=workers_per_node,
            sharding_annotations=self.sharding_annotations,
        )
        self.dp_size = self.sharding_annotations.get_axis_size("data_parallel")
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
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_logprobs",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["pipeline_model_parallel", "tensor_model_parallel"],
            output_is_replicated=["tensor_model_parallel", "pipeline_model_parallel"],
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )
        return logprobs

    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec], micro_batch_size: int = None
    ) -> BatchedDataDict:
        """Get the logprobs of the reference policy for a data dict.

        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.
        Returns: Identical to get_logprobs.
        """
        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_reference_policy_logprobs",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["pipeline_model_parallel", "tensor_model_parallel"],
            output_is_replicated=["tensor_model_parallel", "pipeline_model_parallel"],
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
        batch_size = gbs or self.cfg["train_global_batch_size"]
        # Shard and replicate the batch
        shards = self.dp_size
        sharded_data = data.shard_by_batch_size(shards, batch_size=batch_size)

        # Train each shard in parallel
        futures = self.worker_group.run_all_workers_sharded_data(
            "train",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["pipeline_model_parallel", "tensor_model_parallel"],
            output_is_replicated=["tensor_model_parallel", "pipeline_model_parallel"],
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
        aggregated_results["grad_norm"] = torch.tensor(
            results[0]["grad_norm"], device="cpu"
        )
        # aggregated_results["sums"] = results[0]["sums"]

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
        futures = self.worker_group.run_all_workers_sharded_data(
            "generate",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["pipeline_model_parallel", "tensor_model_parallel"],
            common_kwargs={"greedy": greedy},
            output_is_replicated=["tensor_model_parallel", "pipeline_model_parallel"],
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
            "prepare_for_training", only_on="all_tied_workers"
        )
        ray.get(futures)

    def prepare_for_lp_inference(self, *args, **kwargs):
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference", only_on="all_tied_workers"
        )
        ray.get(futures)

    def finish_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def finish_training(self, *args, **kwargs):
        # Placeholder implementation
        pass

    def prepare_weights_for_ipc(self):
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_weights_for_ipc", only_on="all_tied_workers"
        )
        return ray.get(futures)[0]

    def get_weights_ipc_handles(self, keys):
        """Fetch weight IPC handles from all workers.

        Returns:
            dict: A dictionary mapping device UUIDs to parameter IPC handles.
        """
        # Collect IPC handles from all workers
        worker_handles = ray.get(
            [
                worker.get_weights_ipc_handles.remote(keys=keys)
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
            "offload_before_refit", only_on="all_tied_workers"
        )
        ray.get(futures)

    def offload_after_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_after_refit", only_on="all_tied_workers"
        )
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        offload_to_cpu: bool = True,
    ):
        """Save a checkpoint of the model."""
        futures = self.worker_group.run_all_workers_single_data(
            "save_checkpoint",
            weights_path,
            optimizer_path,
            offload_to_cpu=offload_to_cpu,
            only_on="all_tied_workers",
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
