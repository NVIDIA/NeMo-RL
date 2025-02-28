# `uv` in NeMo-Reinforcer

Using `uv` for Dependency Management in NeMo-Reinforcer

## Overview

`uv` is an incredible tool that simplifies our workflow and is blazingly fast because it's written in Rust. This document outlines why we've adopted `uv` for package management in our repository, particularly for NeMo Reinforcer, and how it helps us manage dependencies across Ray clusters.

## Why `uv`?

### Speed and Efficiency

- Written in Rust, making it significantly faster than traditional Python package managers
- Optimized caching mechanisms that reduce redundant downloads and installations
- Quick environment creation and switching, enabling rapid development cycles

### Isolated Environments

- Creates fully isolated Python environments, preventing dependency conflicts between system packages and project-specific packages
- Avoids nuanced dependency situations where a Python script might accidentally use both virtualenv dependencies and system dependencies
- Ensures consistent behavior across different machines and deployment environments

### Dependency Management in Ray Clusters

- Enables management of heterogeneous Python environments across a Ray cluster
- Provides flexibility for each actor (worker) to use the specific Python dependencies it requires
- Simplifies propagation of environments to worker nodes without manual setup on each node

### Container-Free Flexibility

- Frees us from having to publish many containers for different dependency combinations
- Allows us to define different [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups) and [extras](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies) and select which ones we need dynamically
- Reduces infrastructure complexity and maintenance overhead

## Implementation in NeMo Reinforcer

### Worker Configuration

In our codebase, workers (classes decorated with `@ray.remote`, e.g., `HFPolicyWorker`) define a `DEFAULT_PY_EXECUTABLE` which specifies what dependencies the worker needs. This allows different parts of our application to have their own tailored environments.

### Supported Python Executables

We provide several predefined Python executable configurations in {py:class}`PY_EXECUTABLES <nemo_reinforcer.distributed.virtual_cluster.PY_EXECUTABLES>`:

```python
# --with-editable .: speeds up the install slightly since editable installs don't require full copies
# --cache-dir $UV_CACHE_DIR: caching isn't propagated by default. This will set it if the user has set it.
class PY_EXECUTABLES:
    # This uses the .venv created by `uv`. This is the fastest option, but provides no isolation between workers.
    DEFAULT_VENV = f"{os.environ['VIRTUAL_ENV']}/bin/python"

    # TODO: Debug high run-to-run variance latency with these options
    # Use NeMo-Reinforcer direct dependencies and nothing from system
    DEFAULT = f"uv run --isolated --with-editable . {uv_cache_flag}"
    # Use none of NeMo-Reinforcer's dependencies or the system. Useful for workers that only need standard python packages.
    BARE_BONES = f"uv run --isolated --no-project --with-editable . {uv_cache_flag}"
```

At the moment we **highly recommend** {py:class}`DEFAULT_ENV <nemo_reinforcer.distributed.virtual_cluster.PY_EXECUTABLES.DEFAULT_VENV>` as it results in the fastest bringup of your workload if you are using the `transformers` library and `vllm`.

### Customization

If you need a different Python executable configuration, you can override the default one by passing your own in {py:class}`RayWorkerBuilder.__call__ <nemo_reinforcer.distributed.worker_groups.RayWorkerBuilder.__call__>`. This provides flexibility for special use cases without modifying the core configurations.

## How It Works

When a Ray job is started:

1. The driver process runs in the `uv` environment specified at launch
2. Ray detects this environment and propagates it to worker processes
3. Each worker can specify its own environment through `py_executable` in its runtime environment
4. `uv` efficiently sets up these environments on each worker, using caching to minimize setup time

This approach ensures consistent environments across the cluster while allowing for worker-specific customization when needed.

## Conclusion

Using `uv` for dependency management in NeMo Reinforcer provides us with a fast, flexible, and reliable way to handle Python dependencies across distributed Ray clusters. It eliminates many of the traditional pain points of dependency management in distributed systems while enabling heterogeneous environments that can be tailored to specific workloads.
