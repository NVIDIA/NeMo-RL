# uv in NeMo RL

We use the `uv` Python package installer for managing dependencies in NeMo RL.

## Overview

`uv` is an incredible tool that simplifies our workflow and is blazingly fast because it's written in Rust. This document explains why we've adopted `uv` for package management in our repository, particularly for NeMo RL, and how it helps us manage dependencies across Ray clusters.

## Why `uv`?

### Speed and Efficiency

- Written in Rust, making it significantly faster than traditional Python package managers.
- Optimized caching mechanisms that reduce redundant downloads and installations.
- Quick environment creation and switching, enabling rapid development cycles.

### Isolated Environments

- Creates fully isolated Python environments, preventing dependency conflicts between system packages and project-specific packages.
- Avoids nuanced dependency situations where a Python script might accidentally use both virtualenv dependencies and system dependencies.
- Ensures consistent behavior across different machines and deployment environments.

### Dependency Management in Ray Clusters

- Enables management of heterogeneous Python environments across a Ray cluster.
- Provides flexibility for each actor (worker) to use the specific Python dependencies it requires.
- Simplifies propagation of environments to worker nodes without manual setup on each node.

### Container-Free Flexibility

- Frees us from having to publish many containers for different dependency combinations.
- Allows us to define different [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups) and [extras](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies) and select which ones we need dynamically.
- Reduces infrastructure complexity and maintenance overhead.

## Implementation in NeMo RL

### Worker Configuration

In our codebase, workers (classes decorated with `@ray.remote`, e.g., `HFPolicyWorker`) define a `DEFAULT_PY_EXECUTABLE` which specifies what dependencies the worker needs. This allows different parts of our application to have their own tailored environments.

### Supported Python Executables

We provide several predefined Python executable configurations in {py:class}`PY_EXECUTABLES <nemo_rl.distributed.virtual_cluster.PY_EXECUTABLES>`:

```python
class PY_EXECUTABLES:
    SYSTEM = sys.executable

    # Use NeMo-RL direct dependencies.
    BASE = "uv run --locked"

    # Use NeMo-RL direct dependencies and vllm.
    VLLM = "uv run --locked --extra vllm"
```

To ensure consistent dependencies between actors, we run with `--locked` to make sure the dependencies are consistent with the contents of `uv.lock`.

### Customization

If you need a different Python executable configuration, you can override the default one by passing your own in {py:class}`RayWorkerBuilder.__call__ <nemo_rl.distributed.worker_groups.RayWorkerBuilder.__call__>`. This provides flexibility for special use cases without modifying the core configurations.

## How It Works

When a NeMo RL job is started:

1. The driver script creates several {py:class}`RayWorkerGroup <nemo_rl.distributed.worker_groups.RayWorkerGroup>`s.
2. Each worker group will create their workers which are wrapped in a {py:class}`RayWorkerBuilder <nemo_rl.distributed.worker_groups.RayWorkerBuilder>`
3. Before the worker class is instantiated by the `RayWorkerBuilder`, if (1) `DEFAULT_PY_EXECUTABLE` is defined on the worker class (decorated with `@ray.remote`) and (2) it starts with `uv`; a `venv` is created with all the dependencies it needs and the `runtime_env["py_executable"]` is replaced with the `venv`'s python interpreter.

This approach allows a fast start-up and maintains dependency isolation. It also has the added benefit of having all the virtual environments local under `./venvs`.

## Conclusion

Using `uv` for dependency management in NeMo RL provides us with a fast, flexible, and reliable way to handle Python dependencies across distributed Ray clusters. It eliminates many of the traditional pain points of dependency management in distributed systems, while enabling heterogeneous environments that can be tailored to specific workloads.
