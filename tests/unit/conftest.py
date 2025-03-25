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
from io import StringIO
import time
import pytest
from nemo_reinforcer.utils.logger import GPUMonitoringConfig
from tests import unit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import random
from typing import Callable
import ray

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import random
from typing import Callable
import ray
import json
from nemo_reinforcer.distributed.virtual_cluster import init_ray
from typing import TypedDict
from datetime import datetime

dir_path = os.path.dirname(os.path.abspath(__file__))

UNIT_RESULTS_FILE = os.path.join(dir_path, "unit_results.json")
UNIT_RESULTS_FILE_DATED = os.path.join(
    dir_path, f"unit_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)


class UnitTestData(TypedDict):
    exit_status: int | str
    git_commit: str
    metrics: dict
    gpu_types: list[str]
    coverage: dict


def pytest_sessionstart(session):
    # Delete the unit results file at the start of a new test session
    if os.path.exists(UNIT_RESULTS_FILE):
        try:
            os.remove(UNIT_RESULTS_FILE)
            print(f"Deleted existing results file: {UNIT_RESULTS_FILE}")
        except Exception as e:
            print(f"Warning: Failed to delete results file: {e}")

    # Get the git commit hash
    try:
        import subprocess

        result = subprocess.run(
            ["git", "-C", dir_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_commit = result.stdout.strip()
    except Exception as e:
        git_commit = f"Error getting git commit: {str(e)}"

    session.config._unit_test_data = UnitTestData(
        exit_status="was not set",
        git_commit=git_commit,
        metrics={},
        gpu_types=[],
        coverage={},
    )


@pytest.fixture(scope="session", autouse=True)
def session_data(request, init_ray_cluster):
    """Session-level fixture to store and save metrics data.

    This fixture tracks both metrics from tests and metadata about the test environment.
    The metrics are stored in the 'metrics' dictionary.

    It's set to autouse so that we track metadata and coverage even if no test selected
    explicitly track metrics.
    """
    # Pass init_ray_cluster so that we can access ray metadata

    ############################################################
    # 1. Gather all the unit test data #
    ############################################################
    unit_test_data: UnitTestData = request.config._unit_test_data
    yield unit_test_data

    ############################################################
    # 2. Gather the ray metadata #
    ############################################################
    from nemo_reinforcer.utils.logger import RayGpuMonitorLogger

    logger = RayGpuMonitorLogger(
        collection_interval=float("inf"),
        flush_interval=float("inf"),
        parent_logger=None,
    )
    unit_test_data["gpu_types"] = list(set(logger._collect_gpu_sku().values()))

    ############################################################
    # 3. Gather the coverage data #
    ############################################################
    # We directly access the coverage controller from the plugin manager
    # so we can access the coverage total before the pytest session finishes.
    if request.config.pluginmanager.hasplugin("_cov"):
        plugin = request.config.pluginmanager.getplugin("_cov")
        if plugin.cov_controller:
            cov_controller = plugin.cov_controller
    # We currently don't use the cov_report since we can always access the coverage.json later, but
    # in the future if we want to report the coverage more granularly as part of the session finish,
    # we can access it here.
    cov_report = StringIO()
    cov_total = cov_controller.summary(cov_report)
    unit_test_data["coverage"] = cov_total


@pytest.fixture
def tracker(request, session_data, ray_gpu_monitor):
    """Test-level fixture that automatically captures test function info."""
    # Get fully qualified test name (module::test_function)
    module_name = request.module.__name__
    test_name = request.function.__name__
    qualified_name = f"{module_name}::{test_name}"

    # Initialize an empty dict for this test if it doesn't exist
    if qualified_name not in session_data:
        session_data["metrics"][qualified_name] = {}

    class Tracker:
        def track(self, metric_name: str, value):
            """Tracking an arbitrary metric."""
            session_data["metrics"][qualified_name][metric_name] = value

        def get_max_mem(self):
            metrics = ray_gpu_monitor._collect_metrics()
            max_mem = 0
            for m_name, m_value in metrics.items():
                if m_name.endswith(".memory"):
                    max_mem = max(max_mem, m_value)
            return max_mem

        def log_max_mem(self, metric_name: str):
            session_data["metrics"][qualified_name][metric_name] = self.get_max_mem()

    start_time = time.time()
    yield Tracker()
    end_time = time.time()
    # Prefix with `_` to indicate it's automatically collected
    session_data["metrics"][qualified_name]["_elapsed"] = end_time - start_time


def pytest_sessionfinish(session, exitstatus):
    data = session.config._unit_test_data
    data["exit_status"] = exitstatus
    print(f"\nSaving unit test data to {UNIT_RESULTS_FILE}")
    print(f"and saving to {UNIT_RESULTS_FILE_DATED}")
    with open(UNIT_RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    os.makedirs(os.path.dirname(UNIT_RESULTS_FILE_DATED), exist_ok=True)
    with open(UNIT_RESULTS_FILE_DATED, "w") as f:
        json.dump(data, f, indent=2)


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    """Initialize Ray for the test module and clean up afterward.

    This fixture doesn't need to be called directly.
    """
    init_ray()
    yield
    ray.shutdown()


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor(init_ray_cluster):
    """Initialize Ray for the test module and clean up afterward.

    This fixture doesn't need to be called directly.
    """
    from nemo_reinforcer.utils.logger import RayGpuMonitorLogger

    gpu_monitor = RayGpuMonitorLogger(
        collection_interval=1,
        flush_interval=float("inf"),  # Disabling flushing since we will do it manually
        parent_logger=None,
    )
    gpu_monitor.start()
    yield gpu_monitor
    gpu_monitor.stop()


def _setup_distributed(rank, world_size, port, backend="nccl"):
    """Initialize the distributed environment for a test (internal use only)"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)  # Use the same port for all processes

    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # Set the device for this process
    torch.cuda.set_device(rank)


def _cleanup_distributed():
    """Clean up the distributed environment after a test (internal use only)"""
    dist.destroy_process_group()


@pytest.fixture
def distributed_test_runner():
    """Fixture that returns a function to run distributed tests.

    This fixture provides a reusable way to run a test function across multiple processes
    with PyTorch distributed communication set up.
    """

    def run_distributed_test(
        test_fn: Callable, world_size: int, backend: str = "nccl"
    ) -> None:
        """Run a test function in a distributed environment.

        Args:
            test_fn: The test function to run on each process
            world_size: Number of processes to spawn
            backend: PyTorch distributed backend to use
        """
        # Skip if CUDA is not available and using NCCL backend
        if backend == "nccl" and not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping CUDA-based test")

        # Skip if we don't have enough GPUs for NCCL backend
        if backend == "nccl" and torch.cuda.device_count() < world_size:
            pytest.skip(
                f"Not enough GPUs available. Need {world_size}, got {torch.cuda.device_count()}"
            )

        # Generate a single random port in the main process
        port = random.randint(10000, 20000)

        # Run the test on multiple processes
        mp.spawn(
            _distributed_test_wrapper,
            args=(test_fn, world_size, port, backend),
            nprocs=world_size,
            join=True,
        )

    return run_distributed_test


def _distributed_test_wrapper(
    rank: int, test_fn: Callable, world_size: int, port: int, backend: str
) -> None:
    """Wrapper function that sets up the distributed environment before running the test function.
    Internal use only - use distributed_test_runner fixture instead.

    Args:
        rank: Process rank
        test_fn: The test function to run
        world_size: Total number of processes
        port: Port to use for distributed communication
        backend: PyTorch distributed backend to use
    """
    try:
        # Setup the distributed environment
        _setup_distributed(rank, world_size, port, backend=backend)

        # Run the actual test function
        test_fn(rank, world_size)

        # Clean up
        _cleanup_distributed()
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        _cleanup_distributed()
        raise
