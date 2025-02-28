import pytest
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
from nemo_reinforcer.distributed.virtual_cluster import init_ray


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    """Initialize Ray for the test module and clean up afterward.

    This fixture doesn't need to be called directly.
    """
    init_ray()
    yield
    ray.shutdown()


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
