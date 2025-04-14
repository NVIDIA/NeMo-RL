# Nemo-Reinforcer: A Scalable and Efficient Post-Training Library for Models Ranging from tiny to >100B Parameters, scaling from 1 GPU to 100s

<!-- markdown all in one -->
- [Nemo-Reinforcer: A Scalable and Efficient Post-Training Library for Models Ranging from tiny to \>100B Parameters, scaling from 1 GPU to 100s](#nemo-reinforcer-a-scalable-and-efficient-post-training-library-for-models-ranging-from-tiny-to-100b-parameters-scaling-from-1-gpu-to-100s)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick start](#quick-start)
    - [SFT](#sft)
      - [Single Node](#single-node)
      - [Multi-node](#multi-node)
    - [DPO](#dpo)
      - [Single Node](#single-node-1)
      - [Multi-node](#multi-node-1)
    - [GRPO](#grpo)
      - [Single Node](#single-node-2)
      - [Multi-node](#multi-node-2)
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
- ✅ **DPO Algorithm** - Direct Preference Optimization for alignment
- 🔜 **Larger Model Support** - Native PyTorch support for models up to 70B parameters
- 🔜 **Advanced Parallelism** - FSDP2, TP, SP, and sequence packing for efficient training
- 🔜 **Environment Isolation** - Dependency isolation between components

## Installation

```sh
# For faster setup we use `uv`
pip install uv

# Specify a virtual env that uses Python 3.12
uv venv -p python3.12.9 .venv
# Install NeMo-Reinforcer with vllm
uv pip install -e .[vllm]
# Install NeMo-Reinforcer with dev/test dependencies
uv pip install -e '.[dev,test]'

# Use uv run to launch any runs. 
# Note that it is recommended to not activate the venv and instead use `uv run` since
# it ensures consistent environment usage across different shells and sessions.
# Example: uv run python examples/run_grpo_math.py
```

## Quick start

**Reminder**: Don't forget to set your HF_HOME and WANDB_API_KEY (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

### SFT

We provide a sample SFT experiment that uses the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).

#### Single Node

The default SFT experiment is configured to run on a single GPU. To launch the experiment,

```sh
uv run python examples/run_sft.py
```

This trains `Llama3.2-1B` on one GPU using the SQUAD dataset.

If you have access to more GPUs, you can update the experiment accordingly. To run on 8 GPUs, we update the cluster configuration. We also switch to an 8B Llama base model and increase the batch size:

```sh
uv run python examples/run_sft.py \
  policy.model_name="meta-llama/Meta-Llama-3-8B" \
  policy.train_global_batch_size=128 \
  sft.val_global_batch_size=128 \
  cluster.gpus_per_node=8
```

Refer to [sft.yaml](examples/configs/sft.yaml) for a full list of parameters that can be overridden.

#### Multi-node

For distributed training across multiple nodes:

Set `UV_CACHE_DIR` to a directory that can be read from all workers before running any uv run command.
```sh
export UV_CACHE_DIR=/path/that/all/workers/can/access/uv_cache
```

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=2
# Add a timestamp to make each job name unique
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# SFT experiment uses Llama-3.1-8B model
COMMAND="uv pip install -e .; uv run ./examples/run_sft.py --config examples/configs/sft.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/sft_llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='sft-llama8b'" \
RAY_DEDUP_LOGS=0 \
UV_CACHE_DIR=YOUR_UV_CACHE_DIR \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

### DPO

We provide a sample DPO experiment that uses the [HelpSteer3 dataset](https://huggingface.co/datasets/nvidia/HelpSteer3) for preference-based training.

#### Single Node

The default DPO experiment is configured to run on a single GPU. To launch the experiment:

```sh
uv run python examples/run_dpo.py
```

This trains `Llama3.2-1B-Instruct` on one GPU.

If you have access to more GPUs, you can update the experiment accordingly. To run on 8 GPUs, we update the cluster configuration and switch to an 8B Llama base model:

```sh
uv run python examples/run_dpo.py \
  policy.model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
  policy.train_global_batch_size=128 \
  dpo.val_global_batch_size=128 \
  cluster.gpus_per_node=8
```

Any of the DPO parameters can be customized from the command line. For example:

```sh
uv run python examples/run_dpo.py \
  dpo.sft_loss_weight=0.1 \
  dpo.preference_average_log_probs=True \
  checkpointing.checkpoint_dir="results/llama_dpo_sft" \
  logger.wandb_enabled=True \
  logger.wandb.name="llama-dpo-sft"
```

Refer to [dpo.yaml](examples/configs/dpo.yaml) for a full list of parameters that can be overridden. For an in-depth explanation of how to add your own DPO dataset, refer to the [DPO documentation](docs/guides/dpo.md)

#### Multi-node

For distributed DPO training across multiple nodes:

Set `UV_CACHE_DIR` to a directory that can be read from all workers before running any uv run command.
```sh
export UV_CACHE_DIR=/path/that/all/workers/can/access/uv_cache
```

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=2
# Add a timestamp to make each job name unique
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# DPO experiment uses Llama-3.1-8B model
COMMAND="uv pip install -e .; uv run ./examples/run_dpo.py --config examples/configs/dpo.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/dpo_llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='dpo-llama8b'" \
RAY_DEDUP_LOGS=0 \
UV_CACHE_DIR=YOUR_UV_CACHE_DIR \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

### GRPO

We have a reference GRPO experiment config set up trained for math benchmarks using the [OpenInstructMath2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset.

#### Single Node

To run GRPO on a single GPU for `Llama-3.2-1B-Instruct`:

```sh
# Run the GRPO math example using a 1B parameter model
uv run python examples/run_grpo_math.py
```

By default, this uses the configuration in `examples/configs/grpo_math_1B.yaml`. You can customize parameters with command-line overrides. For example, to run on 8 gpus,

```sh
# Run the GRPO math example using a 1B parameter model using 8 GPUs
uv run python examples/run_grpo_math.py \
  cluster.gpus_per_node=8
```

You can override any of the parameters listed in the yaml configuration file. For example,

```sh
uv run python examples/run_grpo_math.py \
  policy.model_name="Qwen/Qwen2-1.5B" \
  checkpointing.checkpoint_dir="results/qwen1_5b_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="grpo-qwen1_5b_math" \
  logger.num_val_samples_to_print=10 \
```

#### Multi-node

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=2
# Add a timestamp to make each job name unique
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# grpo_math_8b uses Llama-3.1-8B-Instruct model
COMMAND="uv pip install -e .; uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=2 checkpointing.checkpoint_dir='results/llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'" \
RAY_DEDUP_LOGS=0 \
UV_CACHE_DIR=YOUR_UV_CACHE_DIR \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

## Cluster Start

Please visit [Cluster Start](docs/cluster.md) for how to get started on Slurm or Kubernetes.
