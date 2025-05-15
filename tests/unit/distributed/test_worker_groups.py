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
import sys

import pytest
import ray

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup


@ray.remote
class MyTestActor:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.configured_gpus_in_init = kwargs.get("configured_gpus", "not_set")
        self.bundle_indices_seen_in_init = kwargs.get(
            "bundle_indices_seen_in_init", "not_set"
        )
        self.env_vars = dict(os.environ)
        self.pid = os.getpid()

    def get_pid(self):
        return self.pid

    def get_init_args_kwargs(self):
        return self.init_args, self.init_kwargs

    def get_env_var(self, var_name):
        return self.env_vars.get(var_name)

    def echo(self, x):
        return f"Actor {self.pid} echoes: {x}"

    def get_rank_world_size_node_rank_local_rank(self):
        return (
            self.env_vars.get("RANK"),
            self.env_vars.get("WORLD_SIZE"),
            self.env_vars.get("NODE_RANK"),
            self.env_vars.get("LOCAL_RANK"),
        )

    def get_master_addr_port(self):
        return self.env_vars.get("MASTER_ADDR"), self.env_vars.get("MASTER_PORT")

    def check_configured_worker_effect(self):
        return (
            self.configured_gpus_in_init,
            self.bundle_indices_seen_in_init,
            self.env_vars.get("CONFIGURED_WORKER_CALLED"),
        )

    def get_actual_python_executable_path(self):
        return sys.executable

    @staticmethod
    def configure_worker(num_gpus, bundle_indices):
        init_kwargs_update = {
            "configured_gpus": num_gpus,
            "bundle_indices_seen_in_init": bundle_indices is not None,
        }
        resources = {"num_gpus": num_gpus}
        env_vars_update = {"CONFIGURED_WORKER_CALLED": "1"}
        return resources, env_vars_update, init_kwargs_update


MY_TEST_ACTOR_FQN = f"{MyTestActor.__module__}.MyTestActor"


@pytest.fixture(scope="session", autouse=True)
def setup_ray_for_session():
    if not ray.is_initialized():
        # num_gpus=0 because tests don't require actual GPUs and it simplifies setup
        ray.init(num_cpus=4, num_gpus=0, include_dashboard=False)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def register_test_actor(request):
    # Default to PY_EXECUTABLES.SYSTEM if no param is given
    py_exec_to_register = getattr(request, "param", PY_EXECUTABLES.SYSTEM)

    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(MY_TEST_ACTOR_FQN)
    ACTOR_ENVIRONMENT_REGISTRY[MY_TEST_ACTOR_FQN] = py_exec_to_register

    yield MY_TEST_ACTOR_FQN  # Provide the FQN to the test

    # Clean up: revert ACTOR_ENVIRONMENT_REGISTRY to its original state for this FQN
    if MY_TEST_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:  # Check if key still exists
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[MY_TEST_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[MY_TEST_ACTOR_FQN] = original_registry_value


@pytest.fixture
def virtual_cluster():
    # 1 node, 2 CPU bundles. use_gpus=False means num_gpus passed to workers will be 0.
    # bundle_ct_per_node_list=[2] means 1 node with 2 bundles.
    # Since use_gpus=False, these are CPU bundles.
    # master_port_retries is not an explicit arg, it's handled by env var NRL_VIRTUAL_CLUSTER_MAX_RETRIES internally for pg retries.
    # The test's master_port_retries=3 was an assumption, RayVirtualCluster doesn't take it.
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[2], use_gpus=False)
    yield cluster
    cluster.shutdown()


def test_basic_worker_creation_and_method_calls(register_test_actor, virtual_cluster):
    actor_fqn = register_test_actor
    builder = RayWorkerBuilder(actor_fqn)

    # workers_per_node=None should default to one worker per bundle in the cluster (2 bundles = 2 workers)
    worker_group = RayWorkerGroup(
        cluster=virtual_cluster, remote_worker_builder=builder, workers_per_node=None
    )

    assert len(worker_group.workers) == 2, "Should create a worker for each bundle"

    messages = [f"hello from test {i}" for i in range(2)]
    futures = [
        worker.echo.remote(messages[i]) for i, worker in enumerate(worker_group.workers)
    ]
    results = ray.get(futures)

    pids = ray.get([w.get_pid.remote() for w in worker_group.workers])
    assert pids[0] != pids[1], "Actors should be in different processes"

    for i, result in enumerate(results):
        assert f"Actor {pids[i]} echoes: {messages[i]}" == result

    worker_group.shutdown(force=True)


def test_actor_initialization_with_args_kwargs(register_test_actor, virtual_cluster):
    actor_fqn = register_test_actor
    init_args = ("arg1", 123)
    original_init_kwargs = {"kwarg1": "value1", "kwarg2": 456}

    builder = RayWorkerBuilder(actor_fqn, *init_args, **original_init_kwargs)
    # For this test (1 worker, use_gpus=False):
    # num_gpus passed to configure_worker will be 0.
    # bundle_indices will be non-None (e.g., (0, [0])).
    # So, configure_worker adds: {"configured_gpus": 0, "bundle_indices_seen_in_init": True}
    expected_kwargs_from_configure = {
        "configured_gpus": 0,
        "bundle_indices_seen_in_init": True,
    }

    worker_group = RayWorkerGroup(
        cluster=virtual_cluster, remote_worker_builder=builder, workers_per_node=1
    )

    assert len(worker_group.workers) == 1
    worker = worker_group.workers[0]

    ret_args, ret_kwargs = ray.get(worker.get_init_args_kwargs.remote())

    assert ret_args == init_args  # *args are received as a tuple

    # Construct the full expected kwargs dictionary
    expected_final_kwargs = original_init_kwargs.copy()
    expected_final_kwargs.update(expected_kwargs_from_configure)

    assert ret_kwargs == expected_final_kwargs

    worker_group.shutdown(force=True)


def test_environment_variables_setup(register_test_actor, virtual_cluster):
    actor_fqn = register_test_actor
    builder = RayWorkerBuilder(actor_fqn)
    # This will create 2 workers on node 0, with local ranks 0 and 1
    worker_group = RayWorkerGroup(
        cluster=virtual_cluster, remote_worker_builder=builder, workers_per_node=2
    )

    assert len(worker_group.workers) == 2
    world_size = str(len(worker_group.workers))  # "2"

    futures = [
        w.get_rank_world_size_node_rank_local_rank.remote()
        for w in worker_group.workers
    ]
    results = ray.get(futures)

    # Get the master address and port that the worker group used for configuration
    expected_master_addr = worker_group.master_address
    expected_master_port = str(worker_group.master_port)

    for i, worker_results in enumerate(results):
        rank, ws, node_rank, local_rank = worker_results
        assert rank == str(i)
        assert ws == world_size
        assert node_rank == "0"  # Only one node in this cluster
        assert local_rank == str(i)  # Corresponds to bundle_idx

        m_addr, m_port = ray.get(worker_group.workers[i].get_master_addr_port.remote())
        assert m_addr == expected_master_addr
        assert m_port == expected_master_port

    worker_group.shutdown(force=True)


def test_configure_worker_interaction(register_test_actor, virtual_cluster):
    actor_fqn = register_test_actor
    builder = RayWorkerBuilder(actor_fqn)
    # Creates 1 worker. virtual_cluster has use_gpus=False, so num_gpus=0 to builder call.
    worker_group = RayWorkerGroup(
        cluster=virtual_cluster, remote_worker_builder=builder, workers_per_node=1
    )

    assert len(worker_group.workers) == 1
    worker = worker_group.workers[0]

    # MyTestActor.configure_worker receives num_gpus=0 from RayWorkerBuilder
    # (since cluster.use_gpus=False, RayWorkerGroup passes num_gpus=0)
    # bundle_indices for a single worker not in a TP group (local_rank=0) will be (node_idx, [local_bundle_idx])
    # So bundle_indices_seen_in_init should be True.

    configured_gpus, bundle_indices_seen, env_var_set = ray.get(
        worker.check_configured_worker_effect.remote()
    )

    assert configured_gpus == 0  # num_gpus passed to configure_worker
    assert bundle_indices_seen is True  # bundle_indices should be passed
    assert env_var_set == "1"  # Env var from configure_worker

    worker_group.shutdown(force=True)
