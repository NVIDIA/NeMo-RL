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

from nemo_rl.distributed.batched_data_dict import SlicedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
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
        self.stored_data = None
        self.stored_args = None
        self.stored_kwargs = None
        self.call_count = 0

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

    def record_call(self, data=None, *args, **kwargs):
        self.stored_data = data
        self.stored_args = args
        self.stored_kwargs = kwargs
        self.call_count += 1
        return f"Actor {self.pid} called with data: {data}, args: {args}, kwargs: {kwargs}, call_count: {self.call_count}, my_rank: {self.env_vars.get('RANK')}"

    def get_recorded_data(self):
        return self.stored_data, self.stored_args, self.stored_kwargs, self.call_count

    def reset_call_records(self):
        self.stored_data = None
        self.stored_args = None
        self.stored_kwargs = None
        self.call_count = 0

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


@pytest.fixture
def virtual_cluster_4_bundles():
    # 2 nodes, 2 CPU bundles each. use_gpus=False.
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[2, 2], use_gpus=False)
    yield cluster
    cluster.shutdown()


@pytest.fixture
def worker_group_1d_sharding(register_test_actor, virtual_cluster):
    actor_fqn = register_test_actor
    builder = RayWorkerBuilder(actor_fqn)
    # virtual_cluster has 1 node, 2 bundles. Sharding will be across these 2 bundles.
    sharding = NamedSharding(layout=[0, 1], names=["data"])
    worker_group = RayWorkerGroup(
        cluster=virtual_cluster,
        remote_worker_builder=builder,
        workers_per_node=None,  # Should create 2 workers, one per bundle
        sharding_annotations=sharding,
    )
    yield worker_group
    worker_group.shutdown(force=True)


@pytest.fixture
def worker_group_2d_sharding(register_test_actor, virtual_cluster_4_bundles):
    actor_fqn = register_test_actor
    builder = RayWorkerBuilder(actor_fqn)
    # virtual_cluster_4_bundles has 2 nodes, 2 bundles each (4 total bundles).
    # Layout: dp=2, tp=2. Ranks 0,1 on node 0; Ranks 2,3 on node 1.
    sharding = NamedSharding(layout=[[0, 1], [2, 3]], names=["dp", "tp"])
    worker_group = RayWorkerGroup(
        cluster=virtual_cluster_4_bundles,
        remote_worker_builder=builder,
        workers_per_node=None,  # Should create 4 workers, one per bundle
        sharding_annotations=sharding,
    )
    yield worker_group
    worker_group.shutdown(force=True)


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


def test_run_all_workers_single_data_1d_sharding(worker_group_1d_sharding):
    worker_group = worker_group_1d_sharding
    assert len(worker_group.workers) == 2

    # Reset records before call
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    test_data = SlicedDataDict({"key": "value_single"})
    test_arg1 = "arg_single"
    test_kwarg1 = "kwarg_single_val"

    futures = worker_group.run_all_workers_single_data(
        "record_call", test_data, test_arg1, kwarg1=test_kwarg1
    )
    results = ray.get(futures)
    assert len(results) == 2  # Should run on all 2 workers

    for i, worker in enumerate(worker_group.workers):
        data, args, kwargs, count = ray.get(worker.get_recorded_data.remote())
        assert count == 1
        assert data == test_data
        assert args == (test_arg1,)
        assert kwargs == {"kwarg1": test_kwarg1}


def test_run_all_workers_single_data_2d_sharding_no_filter(worker_group_2d_sharding):
    worker_group = worker_group_2d_sharding
    assert len(worker_group.workers) == 4
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    test_data = SlicedDataDict({"key": "value_2d_no_filter"})
    futures = worker_group.run_all_workers_single_data("record_call", test_data)
    results = ray.get(futures)
    assert len(results) == 4  # Runs on all 4 workers

    for worker in worker_group.workers:
        data, _, _, count = ray.get(worker.get_recorded_data.remote())
        assert count == 1
        assert data == test_data


def test_run_all_workers_single_data_2d_sharding_filter_tp(worker_group_2d_sharding):
    worker_group = worker_group_2d_sharding  # dp=2, tp=2
    assert len(worker_group.workers) == 4
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    test_data = SlicedDataDict({"key": "value_2d_filter_tp"})
    # Only run on tp rank 0 for each dp rank
    futures = worker_group.run_all_workers_single_data(
        "record_call", test_data, run_rank_0_only_axes=["tp"]
    )
    results = ray.get(futures)
    assert len(results) == 2  # Runs on 2 workers (dp0-tp0, dp1-tp0)

    # Ranks 0 (dp0,tp0) and 2 (dp1,tp0) should have been called
    # Ranks 1 (dp0,tp1) and 3 (dp1,tp1) should NOT have been called
    expected_called_ranks = [0, 2]
    for i, worker in enumerate(worker_group.workers):
        data, _, _, count = ray.get(worker.get_recorded_data.remote())
        if i in expected_called_ranks:
            assert count == 1
            assert data == test_data
        else:
            assert count == 0
            assert data is None


def test_run_all_workers_single_data_2d_sharding_filter_dp_tp(worker_group_2d_sharding):
    worker_group = worker_group_2d_sharding  # dp=2, tp=2
    assert len(worker_group.workers) == 4
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    test_data = SlicedDataDict({"key": "value_2d_filter_dp_tp"})
    # Only run on dp rank 0 AND tp rank 0
    futures = worker_group.run_all_workers_single_data(
        "record_call", test_data, run_rank_0_only_axes=["dp", "tp"]
    )
    results = ray.get(futures)
    assert len(results) == 1  # Runs on 1 worker (dp0-tp0)

    # Only rank 0 (dp0,tp0) should have been called
    expected_called_ranks = [0]
    for i, worker in enumerate(worker_group.workers):
        data, _, _, count = ray.get(worker.get_recorded_data.remote())
        if i in expected_called_ranks:
            assert count == 1
            assert data == test_data
        else:
            assert count == 0
            assert data is None


def test_run_all_workers_multiple_data_1d_sharding(worker_group_1d_sharding):
    worker_group = worker_group_1d_sharding
    assert len(worker_group.workers) == 2
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    data_for_worker0 = SlicedDataDict({"id": 0, "val": "w0_val"})
    data_for_worker1 = SlicedDataDict({"id": 1, "val": "w1_val"})
    multi_data = [data_for_worker0, data_for_worker1]
    common_arg = "common_arg_multi"

    futures = worker_group.run_all_workers_multiple_data(
        "record_call", multi_data, common_kwargs={"common": common_arg}
    )
    results = ray.get(futures)
    assert len(results) == 2

    # Check worker 0
    d, args, kwargs, count = ray.get(worker_group.workers[0].get_recorded_data.remote())
    assert count == 1
    assert d == data_for_worker0
    assert args == ()
    assert kwargs == {"common": common_arg}

    # Check worker 1
    d, args, kwargs, count = ray.get(worker_group.workers[1].get_recorded_data.remote())
    assert count == 1
    assert d == data_for_worker1
    assert args == ()
    assert kwargs == {"common": common_arg}


def test_run_all_workers_multiple_data_fewer_data_than_workers(
    worker_group_2d_sharding,
):
    worker_group = worker_group_2d_sharding  # 4 workers
    assert len(worker_group.workers) == 4
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    data_for_worker0 = SlicedDataDict({"id": 0})
    data_for_worker1 = SlicedDataDict({"id": 1})
    multi_data = [data_for_worker0, data_for_worker1]  # Only 2 data items

    with pytest.raises(
        AssertionError, match="data length should be equal to the number of workers: "
    ):
        futures = worker_group.run_all_workers_multiple_data("record_call", multi_data)


def test_run_all_workers_sharded_data_1d(worker_group_1d_sharding):
    worker_group = worker_group_1d_sharding  # 2 workers, sharded on "data"
    assert len(worker_group.workers) == 2
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    # Data is a list, sharded along the "data" axis
    sharded_data_input = [
        SlicedDataDict({"shard": 0, "val": "val0"}),
        SlicedDataDict({"shard": 1, "val": "val1"}),
    ]

    future_bundle = worker_group.run_all_workers_sharded_data(
        "record_call", data=sharded_data_input, in_sharded_axes=["data"]
    )
    results = worker_group.get_all_worker_results(future_bundle)
    assert len(results) == 2  # Each worker gets one piece of data

    # Worker 0 gets data[0]
    d0, _, _, c0 = ray.get(worker_group.workers[0].get_recorded_data.remote())
    assert c0 == 1
    assert d0 == sharded_data_input[0]

    # Worker 1 gets data[1]
    d1, _, _, c1 = ray.get(worker_group.workers[1].get_recorded_data.remote())
    assert c1 == 1
    assert d1 == sharded_data_input[1]


def test_run_all_workers_sharded_data_2d_shard_dp_replicate_tp(
    worker_group_2d_sharding,
):
    worker_group = (
        worker_group_2d_sharding  # 4 workers, dp=2, tp=2. layout=[[0,1],[2,3]]
    )
    assert len(worker_group.workers) == 4
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    # Data sharded along dp (2 elements), replicated along tp
    # data_for_dp0 is for workers (0,1) (dp=0, tp=0,1)
    # data_for_dp1 is for workers (2,3) (dp=1, tp=0,1)
    data_for_dp0 = SlicedDataDict({"dp_shard": 0, "content": "dp0_data"})
    data_for_dp1 = SlicedDataDict({"dp_shard": 1, "content": "dp1_data"})
    sharded_data_input = [data_for_dp0, data_for_dp1]

    future_bundle = worker_group.run_all_workers_sharded_data(
        "record_call",
        data=sharded_data_input,
        in_sharded_axes=["dp"],
        replicate_on_axes=["tp"],
    )
    results = worker_group.get_all_worker_results(future_bundle)
    # All 4 workers called, all 4 should return results by default (output_is_replicated not used here)
    assert len(results) == 4

    # Worker 0 (dp0, tp0) gets data_for_dp0
    d0, _, _, c0 = ray.get(worker_group.workers[0].get_recorded_data.remote())
    assert c0 == 1
    assert d0 == data_for_dp0

    # Worker 1 (dp0, tp1) gets data_for_dp0
    d1, _, _, c1 = ray.get(worker_group.workers[1].get_recorded_data.remote())
    assert c1 == 1
    assert d1 == data_for_dp0

    # Worker 2 (dp1, tp0) gets data_for_dp1
    d2, _, _, c2 = ray.get(worker_group.workers[2].get_recorded_data.remote())
    assert c2 == 1
    assert d2 == data_for_dp1

    # Worker 3 (dp1, tp1) gets data_for_dp1
    d3, _, _, c3 = ray.get(worker_group.workers[3].get_recorded_data.remote())
    assert c3 == 1
    assert d3 == data_for_dp1


def test_run_all_workers_sharded_data_2d_free_axis_dp_shard_tp(
    worker_group_2d_sharding,
):
    worker_group = worker_group_2d_sharding  # dp=2, tp=2. layout=[[0,1],[2,3]]
    assert len(worker_group.workers) == 4
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    # Data sharded along tp (2 elements). dp is a free axis (only dp=0 gets data)
    # data_for_tp0 is for worker 0 (dp=0, tp=0)
    # data_for_tp1 is for worker 1 (dp=0, tp=1)
    data_for_tp0 = SlicedDataDict({"tp_shard": 0, "content": "tp0_data"})
    data_for_tp1 = SlicedDataDict({"tp_shard": 1, "content": "tp1_data"})
    sharded_data_input = [data_for_tp0, data_for_tp1]

    future_bundle = worker_group.run_all_workers_sharded_data(
        "record_call", data=sharded_data_input, in_sharded_axes=["tp"]
    )
    results = worker_group.get_all_worker_results(future_bundle)
    # Only workers on dp=0 are called (ranks 0, 1) and return results by default.
    assert len(results) == 2
    assert future_bundle.called_workers == [0, 1]
    assert future_bundle.return_from_workers == [0, 1]

    # Worker 0 (dp0, tp0) gets data_for_tp0
    d0, _, _, c0 = ray.get(worker_group.workers[0].get_recorded_data.remote())
    assert c0 == 1
    assert d0 == data_for_tp0

    # Worker 1 (dp0, tp1) gets data_for_tp1
    d1, _, _, c1 = ray.get(worker_group.workers[1].get_recorded_data.remote())
    assert c1 == 1
    assert d1 == data_for_tp1

    # Worker 2 (dp1, tp0) - not called
    _, _, _, c2 = ray.get(worker_group.workers[2].get_recorded_data.remote())
    assert c2 == 0

    # Worker 3 (dp1, tp1) - not called
    _, _, _, c3 = ray.get(worker_group.workers[3].get_recorded_data.remote())
    assert c3 == 0


def test_run_all_workers_sharded_data_2d_free_axis_dummy_calls(
    worker_group_2d_sharding,
):
    worker_group = worker_group_2d_sharding  # dp=2, tp=2
    assert len(worker_group.workers) == 4
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    data_for_tp0 = SlicedDataDict({"tp_shard": 0})
    data_for_tp1 = SlicedDataDict({"tp_shard": 1})
    sharded_data_input = [data_for_tp0, data_for_tp1]

    future_bundle = worker_group.run_all_workers_sharded_data(
        "record_call",
        data=sharded_data_input,
        in_sharded_axes=["tp"],  # dp is free axis
        make_dummy_calls_to_free_axes=True,
    )
    results = worker_group.get_all_worker_results(future_bundle)
    # dp=0 workers (0,1) get data. dp=1 workers (2,3) get None.
    # All 4 called, by default all 4 results returned unless output_is_replicated is set.
    # However, return_from_workers will only include those that are not rank > 0 on free axes if output_is_replicated is not set.
    assert len(results) == 2  # Only from dp=0 workers
    assert sorted(future_bundle.called_workers) == [0, 1, 2, 3]
    assert sorted(future_bundle.return_from_workers) == [
        0,
        1,
    ]  # Only dp=0 workers results returned

    d0, _, _, c0 = ray.get(
        worker_group.workers[0].get_recorded_data.remote()
    )  # dp0, tp0
    assert c0 == 1
    assert d0 == data_for_tp0

    d1, _, _, c1 = ray.get(
        worker_group.workers[1].get_recorded_data.remote()
    )  # dp0, tp1
    assert c1 == 1
    assert d1 == data_for_tp1

    d2, _, _, c2 = ray.get(
        worker_group.workers[2].get_recorded_data.remote()
    )  # dp1, tp0 (dummy)
    assert c2 == 1
    assert d2 is None  # Dummy call

    d3, _, _, c3 = ray.get(
        worker_group.workers[3].get_recorded_data.remote()
    )  # dp1, tp1 (dummy)
    assert c3 == 1
    assert d3 is None  # Dummy call


def test_run_all_workers_sharded_data_2d_output_replicated(worker_group_2d_sharding):
    worker_group = worker_group_2d_sharding  # dp=2, tp=2. layout=[[0,1],[2,3]]
    assert len(worker_group.workers) == 4
    ray.get([w.reset_call_records.remote() for w in worker_group.workers])

    # Data sharded on dp, replicated on tp. Output is also replicated on tp.
    data_dp0 = SlicedDataDict({"dp": 0})
    data_dp1 = SlicedDataDict({"dp": 1})
    sharded_input = [data_dp0, data_dp1]

    future_bundle = worker_group.run_all_workers_sharded_data(
        "record_call",
        data=sharded_input,
        in_sharded_axes=["dp"],
        replicate_on_axes=["tp"],
        output_is_replicated=["tp"],  # Only return from tp=0 for each dp group
    )
    results = worker_group.get_all_worker_results(future_bundle)
    # All 4 workers called, but only 2 results returned (dp0-tp0, dp1-tp0)
    assert len(results) == 2
    assert sorted(future_bundle.called_workers) == [0, 1, 2, 3]
    assert sorted(future_bundle.return_from_workers) == [0, 2]

    # Check calls
    d0, _, _, c0 = ray.get(
        worker_group.workers[0].get_recorded_data.remote()
    )  # dp0,tp0
    assert c0 == 1
    assert d0 == data_dp0
    d1, _, _, c1 = ray.get(
        worker_group.workers[1].get_recorded_data.remote()
    )  # dp0,tp1
    assert c1 == 1
    assert d1 == data_dp0
    d2, _, _, c2 = ray.get(
        worker_group.workers[2].get_recorded_data.remote()
    )  # dp1,tp0
    assert c2 == 1
    assert d2 == data_dp1
    d3, _, _, c3 = ray.get(
        worker_group.workers[3].get_recorded_data.remote()
    )  # dp1,tp1
    assert c3 == 1
    assert d3 == data_dp1

    # Check results (should be from worker 0 and worker 2)
    # Assuming MultiWorkerFuture.get_results returns in order of return_from_workers
    assert "my_rank: 0" in results[0]  # from worker 0
    assert "my_rank: 2" in results[1]  # from worker 2
