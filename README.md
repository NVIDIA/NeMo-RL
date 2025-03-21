# Nemo-Reinforcer: A Scalable and Efficient Post-Training Library for Models Ranging from 1 GPU to 1000s, and from Tiny to >100B Parameters

<!-- markdown all in one -->
- [Nemo-Reinforcer: A Scalable and Efficient Post-Training Library for Models Ranging from 1 GPU to 1000s, and from Tiny to \>100B Parameters](#nemo-reinforcer-a-scalable-and-efficient-post-training-library-for-models-ranging-from-1-gpu-to-1000s-and-from-tiny-to-100b-parameters)
  - [Features](#features)
  - [Installation](#installation)
  - [Cluster Start](#cluster-start)

**Nemo-Reinforcer** is a scalable and efficient post-training library designed for models ranging from 1 GPU to thousands, and from tiny to over 100 billion parameters.

What you can expect:

- **Seamless integration with HuggingFace** for ease of use, allowing users to leverage a wide range of pre-trained models and tools.
- **High-performance implementation with Megatron core**, supporting various parallelism techniques for large models (>100B) and large context lengths.
- **Efficient resource management using Ray**, enabling scalable and flexible deployment across different hardware configurations.
- **Flexibility** with a modular design that allows easy integration and customization.
- **Comprehensive documentation** that is both detailed and user-friendly, with practical examples.

## Features

_✅ Available now | 🔜 Coming in v0.2_

- ✅ **Fast Generation** - vLLM backend for optimized inference
- ✅ **HuggingFace Integration** - Works with 1-8B models (Qwen1.5, Llama)
- ✅ **Distributed Training** - FSDP support and Ray-based infrastructure
- ✅ **Environment Support** - Support for multi-environment training.
- ✅ **Learning Algorithms** - GRPO (Group Relative Policy Optimization) and SFT (Supervised Fine-Tuning)
- ✅ **Worker Isolation** - Process isolation between RL Actors (no worries about global state)
- 🔜 **Larger Model Support** - Native PyTorch support for models up to 70B parameters
- 🔜 **Advanced Parallelism** - FSDP2, TP, SP, and sequence packing for efficient training
- 🔜 **Environment Isolation** - Dependency isolation between components
- 🔜 **DPO Algorithm** - Direct Preference Optimization for alignment

## Installation

```sh
# For faster setup we use `uv`
pip install uv

# Specify a virtual env that uses Python 3.12
uv venv -p python3.12.9 .venv
# Install NeMo-Reinforcer with vllm
uv pip install -e .
# Install NeMo-Reinforcer with dev/test dependencies
uv pip install -e '.[dev,test]'

# Use uv run to launch any runs. 
# Note that it is recommended to not activate the venv and instead use `uv run` since
# it ensures consistent environment usage across different shells and sessions.
uv run python examples/run_grpo_math.py
```

## Cluster Start

Please visit [Cluster Start](docs/cluster.md) for how to get started on Slurm or Kubernetes.
