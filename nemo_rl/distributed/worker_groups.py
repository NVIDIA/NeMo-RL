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
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.batched_data_dict import SlicedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.utils.venvs import create_local_venv


@dataclass
class MultiWorkerFuture:
    """Container for Ray futures with associated worker information."""

    futures: List[ray.ObjectRef]
    used_workers: List[int]
    respect_tied_workers: bool = True

    def get_results(self, worker_group):
        """Get results from the futures, optionally respecting tied workers.

        When respect_tied_workers is True, this method deduplicates results by returning
        only one result per tied worker group.

        The method uses worker_group.worker_to_tied_group_index to identify which tied
        worker group each worker belongs to, then selects only the first result from each group.

        Args:
            worker_group: The RayWorkerGroup that created this bundle

        Returns:
            List of results, deduplicated by tied workers if respect_tied_workers is True
        """
        # Basic case: Get all results
        all_results = ray.get(self.futures)

        # If we don't need to deduplicate by tied workers, return all results
        if not self.respect_tied_workers:
            return all_results

        if not self.used_workers:
            return all_results

        # Create tied worker sets based on used workers
        active_tied_workers = {}
        for i, worker_idx in enumerate(self.used_workers):
            tied_worker_idx = worker_group.worker_to_tied_group_index.get(worker_idx)
            if tied_worker_idx is None:
                continue

            if tied_worker_idx not in active_tied_workers:
                active_tied_workers[tied_worker_idx] = []
            active_tied_workers[tied_worker_idx].append(i)

        # Take the first result from each tied worker group
        tied_worker_results = []
        for tied_worker_idx in sorted(active_tied_workers.keys()):
            if active_tied_workers[tied_worker_idx]:
                result_idx = active_tied_workers[tied_worker_idx][0]
                tied_worker_results.append(all_results[result_idx])

        return tied_worker_results


class RayWorkerBuilder:
    @ray.remote
    class IsolatedWorkerInitializer:
        def __init__(self, ray_actor_class_fqn: str, *init_args, **init_kwargs):
            self.ray_actor_class_fqn = ray_actor_class_fqn
            self.init_args = init_args
            self.init_kwargs = init_kwargs

        def create_worker(
            self,
            placement_group: PlacementGroup,
            placement_group_bundle_index: int,
            num_gpus: int,
            bundle_indices: Optional[tuple] = None,
            **extra_options: Optional[Dict[str, Any]],
        ):
            """Create a Ray worker with the specified configuration.

            Order of precedence for worker options configuration (from lowest to highest):
            1. Options passed by the user to __call__ (extra_options)
            2. Options required by the worker via configure_worker (may override user options with warning)
            3. Options set by the RayWorkerBuilder.__call__ (specifically scheduling strategy)

            If the worker needs to override user-provided options, it should log a warning
            to inform the user about the change and the reason for it.

            Args:
                placement_group: Ray placement group for resource allocation
                placement_group_bundle_index: Index of the bundle in the placement group
                num_gpus: Number of GPUs to allocate to this worker
                bundle_indices: Tuple of (node_idx, local_bundle_indices) for tensor parallelism (if applicable)
                extra_options: Additional options to pass to the Ray actor (may be overridden by actor's configure_worker(...) method)

            Returns:
                A Ray actor reference to the created worker
            """
            # Set up worker arguments and resources
            module_name, class_name = self.ray_actor_class_fqn.rsplit(".", 1)
            module = importlib.import_module(module_name)
            worker_class = getattr(module, class_name)
            worker_kwargs = dict(self.init_kwargs)
            options = deepcopy(extra_options)

            # Use the worker's configuration interface if available
            if hasattr(worker_class, "configure_worker"):
                # Get complete worker configuration from the worker class
                resources, env_vars, init_kwargs = worker_class.configure_worker(
                    num_gpus=num_gpus,
                    bundle_indices=bundle_indices,
                )

                # Apply resource configuration
                if resources and "num_gpus" in resources:
                    num_gpus = resources["num_gpus"]

                # Apply environment variables if provided
                if env_vars:
                    if "runtime_env" not in options:
                        options["runtime_env"] = {}
                    for k, v in env_vars.items():
                        options["runtime_env"]["env_vars"][k] = v

                # Apply initialization parameters
                if init_kwargs:
                    worker_kwargs.update(init_kwargs)

            # Create options for Ray actor
            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=placement_group_bundle_index,
                placement_group_capture_child_tasks=True,
            )
            options["num_gpus"] = num_gpus
            # If the user hasn't specified a py_executable, use the worker class's default
            if not options.get("runtime_env", {}).get("py_executable", None):
                if "runtime_env" not in options:
                    options["runtime_env"] = {}
                options["runtime_env"]["py_executable"] = get_actor_python_env(
                    self.ray_actor_class_fqn
                )

            if (
                options.get("runtime_env", {})
                .get("py_executable", "n/a")
                .startswith("uv")
            ):
                # If the py_executable begins with uv it signals that we need to create a
                #  local venv first and then replace the py_executable with the local venv's python.
                #  The directory the venv will be created in is controlled by the env var
                #  NEMO_RL_VENV_DIR and defaults to $GIT_ROOT/venvs/.
                unwrapped_cls = worker_class.__ray_actor_class__
                venv_python = create_local_venv(
                    py_executable=options["runtime_env"]["py_executable"],
                    venv_name=f"{unwrapped_cls.__module__}.{unwrapped_cls.__name__}",
                )
                options["runtime_env"]["py_executable"] = venv_python
            worker = worker_class.options(**options).remote(
                *self.init_args, **worker_kwargs
            )
            return worker

    def __init__(self, ray_actor_class_fqn: str, *args, **kwargs):
        self.ray_actor_class_fqn = ray_actor_class_fqn
        self.args = args
        self.kwargs = kwargs

    def __call__(
        self,
        placement_group: PlacementGroup,
        placement_group_bundle_index: int,
        num_gpus: int,
        bundle_indices: Optional[tuple] = None,
        **extra_options: Dict[str, Any],
    ):
        """Create a Ray worker with the specified configuration.

        Order of precedence for worker options configuration (from lowest to highest):
        1. Options passed by the user to __call__ (extra_options)
        2. Options required by the worker via configure_worker (may override user options with warning)
        3. Options set by the RayWorkerBuilder.__call__ (specifically scheduling strategy)

        If the worker needs to override user-provided options, it should log a warning
        to inform the user about the change and the reason for it.

        Args:
            placement_group: Ray placement group for resource allocation
            placement_group_bundle_index: Index of the bundle in the placement group
            num_gpus: Number of GPUs to allocate to this worker
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for tensor parallelism (if applicable)
            extra_options: Additional options to pass to the Ray actor (may be overridden by actor's configure_worker(...) method)

        Returns:
            A Ray actor reference to the created worker
        """
        # Set up worker arguments and resources
        options = deepcopy(extra_options)

        # If the user hasn't specified a py_executable, use the worker class's default
        initializer_options = {}
        if not options.get("runtime_env", {}).get("py_executable", None):
            if "runtime_env" not in options:
                options["runtime_env"] = {}
            options["runtime_env"]["py_executable"] = get_actor_python_env(
                self.ray_actor_class_fqn
            )

        if options.get("runtime_env", {}).get("py_executable", "n/a").startswith("uv"):
            # If the py_executable begins with uv it signals that we need to create a
            #  local venv first and then replace the py_executable with the local venv's python.
            #  The directory the venv will be created in is controlled by the env var
            #  NEMO_RL_VENV_DIR and defaults to $GIT_ROOT/venvs/.
            # unwrapped_cls = worker_class.__ray_actor_class__
            venv_python = create_local_venv(
                py_executable=options["runtime_env"]["py_executable"],
                venv_name=self.ray_actor_class_fqn,
            )
            options["runtime_env"]["py_executable"] = venv_python

        initializer_options = {"runtime_env": options["runtime_env"]}
        isolated_initializer = self.IsolatedWorkerInitializer.options(
            **initializer_options
        ).remote(self.ray_actor_class_fqn, *self.args, **self.kwargs)
        worker = ray.get(
            isolated_initializer.create_worker.remote(
                placement_group,
                placement_group_bundle_index,
                num_gpus,
                bundle_indices,
                **options,
            )
        )
        # We hold onto a reference to the initializer actor to avoid gc (would kill the child, 'real' actor)
        worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC = isolated_initializer
        return worker


class RayWorkerGroup:
    """Manages a group of distributed Ray worker/actor processes that execute tasks in parallel.

    This class creates and manages Ray actor instances that run on resources
    allocated by a RayVirtualCluster. It handles:
    - Worker creation and placement on specific GPU resources
    - Setting up distributed training environment variables (rank, world size, etc.)
    - Executing methods across all workers in parallel
    - Collecting and aggregating results
    - Support for tied worker groups where multiple workers process the same data
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        remote_worker_builder: RayWorkerBuilder,
        workers_per_node: Optional[Union[int, List[int]]] = None,
        name_prefix: str = "",
        bundle_indices_list: Optional[List[tuple]] = None,
    ):
        """Initialize a group of distributed Ray workers.

        Args:
            cluster: RayVirtualCluster
            remote_worker_builder: Callable that launches a ray worker and has updatable options
            workers_per_node: Defaults to launch one worker per bundle in the cluster.
                          Alternatively specify an int or list to launch a different number of workers per node.
            name_prefix: Optional prefix for the names of the workers
            bundle_indices_list: Explicit list of (node_idx, [local_bundle_indices]) tuples.
                               Each tuple defines a tied group of workers placed on the same node.
                               If provided, workers_per_node is ignored.
        """
        self._workers = []
        self._worker_metadata = []
        self.cluster = cluster
        self.name_prefix = name_prefix
        self.tied_workers_groups = []
        # Maps worker indices to their corresponding tied group index
        # For example, if worker with index 3 belongs to tied worker group 1,
        # then worker_to_tied_group_index[3] = 1
        self.worker_to_tied_group_index = {}

        # If explicit bundle indices are provided, use those
        if bundle_indices_list is None:
            # Create bundle_indices_list from workers_per_node specification
            # In this case, each worker is its own group (no tied workers)
            bundle_indices_list = []

            # Determine how many workers per node
            if workers_per_node is None:
                workers_per_node = [
                    pg.bundle_count for pg in self.cluster.get_placement_groups()
                ]
            elif isinstance(workers_per_node, int):
                workers_per_node = [workers_per_node] * self.cluster.node_count()
            elif not isinstance(workers_per_node, list):
                raise ValueError(
                    "workers_per_node must be None(for default node distribution), an int, or a list"
                )

            # Validate workers_per_node
            assert len(workers_per_node) == self.cluster.node_count(), (
                "workers_per_node_list must be the same length as the number of nodes in the virtual cluster"
            )
            assert all(
                [
                    workers_per_node[i] <= pg.bundle_count
                    for i, pg in enumerate(self.cluster.get_placement_groups())
                ]
            ), (
                "workers_per_node must be less than or equal to the number of bundles in the placement groups"
            )

            # Create bundle_indices_list where each worker is its own group
            for node_idx, worker_count in enumerate(workers_per_node):
                for local_idx in range(worker_count):
                    # Each worker is its own single-element group
                    bundle_indices_list.append((node_idx, [local_idx]))

        # Create workers based on the bundle_indices_list
        self._create_workers_from_bundle_indices(
            remote_worker_builder, bundle_indices_list
        )

    def _create_workers_from_bundle_indices(
        self, remote_worker_builder, bundle_indices_list
    ):
        """Create workers based on explicit bundle indices for tied worker groups.

        Args:
            remote_worker_builder: Builder function for Ray actors
            bundle_indices_list: List of (node_idx, local_bundle_indices) tuples, where each tuple
                               specifies a tied group with its node and local bundle indices.
        """
        self.master_address, self.master_port = (
            self.cluster.get_master_address_and_port()
        )

        # Count total workers
        self.world_size = sum(len(indices) for _, indices in bundle_indices_list)
        global_rank = 0

        for group_idx, (node_idx, local_bundle_indices) in enumerate(
            bundle_indices_list
        ):
            current_group = []

            # Get the placement group for this node
            pg = self.cluster.get_placement_groups()[node_idx]
            is_tp_group = len(local_bundle_indices) > 1

            for local_rank, bundle_idx in enumerate(local_bundle_indices):
                # Set up basic distributed environment variables
                env_vars = dict(
                    os.environ
                )  # Pass thru all user environment variables (at the lowest precendence)
                env_vars.update(
                    {
                        "RANK": str(global_rank),
                        "LOCAL_RANK": str(bundle_idx),
                        "WORLD_SIZE": str(self.world_size),
                        "MASTER_ADDR": self.master_address,
                        "MASTER_PORT": str(self.master_port),
                        "NODE_RANK": str(node_idx),
                    }
                )

                # For tensor parallel groups, only the first worker gets bundle_indices
                worker_bundle_indices = (
                    (node_idx, local_bundle_indices) if local_rank == 0 else None
                )

                # Create a descriptive name based on group structure
                name = (
                    f"{self.name_prefix}-grp{group_idx}-{local_rank}"
                    if is_tp_group
                    else f"{self.name_prefix}-{node_idx}-{bundle_idx}"
                )

                # Calculate GPU resources
                num_gpus = (
                    1 / self.cluster.max_colocated_worker_groups
                    if self.cluster.use_gpus
                    else 0
                )

                # Pass these options to the remote_worker_builder
                runtime_env = {"env_vars": env_vars}
                extra_options = {"runtime_env": runtime_env, "name": name}

                # Create the worker
                worker = remote_worker_builder(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    num_gpus=num_gpus,
                    bundle_indices=worker_bundle_indices,
                    **extra_options,
                )

                # Store worker metadata
                worker_idx = len(self._workers)
                current_group.append(worker_idx)
                self.worker_to_tied_group_index[worker_idx] = group_idx
                self._workers.append(worker)
                self._worker_metadata.append(
                    {
                        "node_idx": node_idx,
                        "local_rank": local_rank,
                        "global_rank": global_rank,
                        "name": name,
                        "bundle_indices": worker_bundle_indices,
                        "tied_group_idx": group_idx,
                    }
                )

                global_rank += 1

            # Add this tied group to our list
            self.tied_workers_groups.append(current_group)

    @property
    def workers(self):
        return self._workers

    @property
    def worker_metadata(self):
        return self._worker_metadata

    @property
    def group_count(self):
        """Number of tied worker groups."""
        return len(self.tied_workers_groups)

    def run_all_workers_multiple_data(
        self,
        method_name: str,
        data: List[SlicedDataDict],
        common_kwargs: Optional[Dict[str, Any]] = None,
        only_on: Literal["all", "tied_leader", "all_tied_workers"] = "all",
    ):
        """Run a method on all workers in parallel with different data.

        Args:
            method_name: Name of the method to call on each worker
            data: List of data slices to pass to workers/groups
            common_kwargs: Additional keyword arguments to pass to all workers
            only_on: Determines which workers receive data and execute the method:
                    - "all": Each worker gets its own data slice
                    - "tied_leader": Only the first worker in each tied group receives data
                    - "all_tied_workers": All workers in each tied group receive the same data slice

        Returns:
            MultiWorkerFuture: Object containing futures and their associated worker information
        """
        # Verify that the data is a list of SlicedDataDict objects
        if not all(isinstance(d, SlicedDataDict) for d in data):
            warnings.warn(
                f"Expected all elements in 'data' to be of type SlicedDataDict, but got "
                f"{[type(d).__name__ for d in data]}. This may cause unexpected behavior. "
                f"Please use make sure you're passing in Sharded Data to this function (and not replicated data)",
                UserWarning,
            )

        if common_kwargs is None:
            common_kwargs = {}

        futures = []
        used_workers = []

        respect_tied_workers = only_on in {"tied_leader", "all_tied_workers"}

        if only_on == "all":
            # Regular case - each worker gets its own data slice
            for worker_id, worker in enumerate(self.workers):
                if worker_id >= len(data):
                    break
                method = getattr(worker, method_name)
                futures.append(method.remote(data[worker_id], **common_kwargs))
                used_workers.append(worker_id)

        elif respect_tied_workers:
            # If there are fewer data slices than tied worker groups, use only the first N tied worker groups
            active_tied_worker_count = min(len(data), len(self.tied_workers_groups))
            if active_tied_worker_count < len(self.tied_workers_groups):
                print(
                    f"Warning: Using only {active_tied_worker_count} of {len(self.tied_workers_groups)} tied worker groups due to limited data slices"
                )

            # For each tied worker group, all workers in the group get the same data slice
            for tied_worker_idx in range(active_tied_worker_count):
                tied_worker_group = self.tied_workers_groups[tied_worker_idx]
                tied_worker_data = data[tied_worker_idx]

                if only_on == "all_tied_workers":
                    # Running on all workers in the non-vllm case
                    for worker_idx in tied_worker_group:
                        futures.append(
                            getattr(self._workers[worker_idx], method_name).remote(
                                tied_worker_data, **common_kwargs
                            )
                        )

                        used_workers.append(worker_idx)
                else:
                    # Running only on the leader of the tied worker group for vllm case
                    futures.append(
                        getattr(
                            self._workers[tied_worker_group[0]], method_name
                        ).remote(tied_worker_data, **common_kwargs)
                    )
                    used_workers.append(tied_worker_group[0])
        else:
            raise ValueError(f"Invalid value for only_on: {only_on}")

        # Return a MultiWorkerFuture containing both futures and worker information
        return MultiWorkerFuture(
            futures=futures,
            used_workers=used_workers,
            respect_tied_workers=respect_tied_workers,
        )

    def run_all_workers_single_data(
        self,
        method_name: str,
        *args,
        only_on: Literal["all", "tied_leader", "all_tied_workers"] = "all",
        **kwargs,
    ):
        """Run a method on all workers in parallel with the same data.

        Args:
            method_name: Name of the method to call on each worker
            only_on: Determines which workers to run the method on:
                    - "all": Run on all workers
                    - "tied_leader": Run only on the first worker of each tied worker group
                    - "all_tied_workers": Run on all workers in each tied worker group
            *args, **kwargs: Arguments to pass to the method

        Returns:
            List[ray.ObjectRef]: A list of ray futures
        """
        futures = []

        respect_tied_workers = only_on in {"tied_leader", "all_tied_workers"}

        if only_on == "all":
            for worker in self.workers:
                method = getattr(worker, method_name)
                futures.append(method.remote(*args, **kwargs))
        elif respect_tied_workers:
            for tied_worker_group in self.tied_workers_groups:
                if only_on == "all_tied_workers":
                    # Running on all workers in the non-vllm case
                    for worker_idx in tied_worker_group:
                        futures.append(
                            getattr(self._workers[worker_idx], method_name).remote(
                                *args, **kwargs
                            )
                        )
                else:
                    futures.append(
                        getattr(
                            self._workers[tied_worker_group[0]], method_name
                        ).remote(*args, **kwargs)
                    )
        else:
            raise ValueError(f"Invalid value for only_on: {only_on}")

        return futures

    def get_all_worker_results(self, future_bundle):
        """Get results from all workers, optionally filtering to get just one result per tied worker group.

        Args:
            future_bundle: MultiWorkerFuture containing futures and worker information.
                          When future_bundle.respect_tied_workers is True, only results from
                          the leaders of tied worker groups are returned.

        Returns:
            List of results, deduplicated as specified in the future_bundle
        """
        return future_bundle.get_results(self)

    def shutdown(
        self,
        cleanup_method: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        force: bool = False,
    ):
        """Shutdown all workers in the worker group.

        Args:
            cleanup_method: Optional method name to call on each worker before termination.
                            If provided, this method will be called on each worker to allow
                            for graceful cleanup.
            timeout: Timeout in seconds for graceful shutdown. Only applicable if cleanup_method is provided.
                     If None, wait indefinitely for workers to complete their cleanup.
            force: If True, forcefully terminate workers with ray.kill() even if cleanup_method is provided.
                   If cleanup_method is None, workers are always forcefully terminated.

        Returns:
            bool: True if all workers were successfully shut down
        """
        if not self._workers:
            return True

        success = True

        # First attempt graceful shutdown if cleanup method is provided and force=False
        if cleanup_method is not None and not force:
            try:
                # Call cleanup method on all workers
                futures = self.run_all_workers_single_data(cleanup_method)

                # Wait for all cleanup operations to complete with timeout
                if timeout is not None:
                    ray.get(futures, timeout=timeout)
                else:
                    ray.get(futures)

            except (ray.exceptions.RayTaskError, ray.exceptions.GetTimeoutError) as e:
                success = False
                print(
                    f"Error during graceful shutdown: {e}. Falling back to force termination."
                )
                force = True

        # Force kill any remaining workers
        if force or cleanup_method is None:
            for worker in self._workers:
                try:
                    ray.kill(worker)
                except Exception as e:
                    success = False
                    print(f"Error killing worker: {e}")

        # Clear worker lists
        self._workers = []
        self._worker_metadata = []
        self.tied_workers_groups = []
        self.worker_to_tied_group_index = {}

        return success

    def print_worker_layout(self):
        """Prints a visual representation of the worker layout across the virtual cluster.

        This shows which workers are assigned to which nodes and GPUs.
        """
        self.cluster.print_cluster_grid(self)
