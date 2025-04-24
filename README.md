# Nemo-Reinforcer: A Scalable and Efficient Post-Training Library for Models Ranging from tiny to >100B Parameters, scaling from 1 GPU to 100s

<!-- markdown all in one -->
- [Nemo-Reinforcer: A Scalable and Efficient Post-Training Library for Models Ranging from tiny to \>100B Parameters, scaling from 1 GPU to 100s](#nemo-reinforcer-a-scalable-and-efficient-post-training-library-for-models-ranging-from-tiny-to-100b-parameters-scaling-from-1-gpu-to-100s)
  - [Features](#features)
  - [Prerequisuites](#prerequisuites)
  - [Quick start](#quick-start)
    - [GRPO](#grpo)
      - [Single Node](#single-node)
      - [Multi-node](#multi-node)
        - [GRPO Qwen2.5-32B](#grpo-qwen25-32b)
    - [SFT](#sft)
      - [Single Node](#single-node-1)
      - [Multi-node](#multi-node-1)
    - [DPO](#dpo)
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

✅ _Available now_ | 🔜 _Coming in v0.3_

- ✅ **Fast Generation** - vLLM backend for optimized inference
- ✅ **HuggingFace Integration** - Works with 1-32B models (Qwen2.5, Llama)
- ✅ **Distributed Training** - FSDP support and Ray-based infrastructure
- ✅ **Environment Support** - Support for multi-environment training.
- ✅ **Learning Algorithms** - GRPO (Group Relative Policy Optimization), SFT (Supervised Fine-Tuning), and DPO (Direct Preference Optimization)
- ✅ **Multi-Turn RL** - multi-turn generation and training for RL with tool use, games, etc. 
- ✅ **Large Model Support** - Native PyTorch support for models up to 32B parameters
- ✅ **Advanced Parallelism** - FSDP2, TP, and SP for efficient training
- ✅ **Worker Isolation** - Process isolation between RL Actors (no worries about global state)
- ✅ **Environment Isolation** - Dependency isolation between components

- 🔜 **(Even) Larger Model Support** - Native PyTorch & Megatron
- 🔜 **Improved Native Performance** - Improve training time for Native Pytorch Models
- 🔜 **Megatron Policy** - Support advanced parallelism in training with Megatron Core
- 🔜 **Megatron Inference** - Support Megatron Inference for day-0 support for new megatron models
- 🔜 **MoE Models** - Support DeepseekV3 and Llama4

## Prerequisuites

```sh
# For faster setup and environment isolation, we use `uv`
pip install uv

# If you cannot install at the system level, you can install for your user with
# pip install --user uv

# Use `uv run` to launch all commands. It handles pip installing implicitly and
# ensures your environment is up to date with our lock file.

# Note that it is not recommended to activate the venv and instead use `uv run` since
# it ensures consistent environment usage across different shells and sessions.
# Example: uv run python examples/run_grpo_math.py
```

## Quick start

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

### GRPO

We have a reference GRPO experiment config set up trained for math benchmarks using the [OpenInstructMath2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset.

#### Single Node

To run GRPO on a single GPU for `Qwen/Qwen2.5-1.5B`:

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
  policy.model_name="Llama-3.2-1B-Instruct" \
  checkpointing.checkpoint_dir="results/llama1b_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="grpo-llama1b_math" \
  logger.num_val_samples_to_print=10 \
```

#### Multi-node

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=2

# grpo_math_8b uses Llama-3.1-8B-Instruct model
COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=2 checkpointing.checkpoint_dir='results/llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'" \
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

##### GRPO Qwen2.5-32B

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=16

# Download Qwen before the job starts to avoid spending time downloading during the training loop
HF_HOME=/path/to/hf_home huggingface-cli download Qwen/Qwen2.5-32B

# Ensure HF_HOME is included in your MOUNTS
HF_HOME=/path/to/hf_home \
COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml policy.model_name='Qwen/Qwen2.5-32B' policy.generation.vllm_cfg.tensor_parallel_size=4 policy.max_total_sequence_length=16384 cluster.num_nodes=${NUM_ACTOR_NODES} policy.dtensor_cfg.enabled=True policy.dtensor_cfg.tensor_parallel_size=8 policy.dtensor_cfg.sequence_parallel=True policy.dtensor_cfg.activation_checkpointing=True checkpointing.checkpoint_dir='results/qwen2.5-32b' logger.wandb_enabled=True logger.wandb.name='qwen2.5-32b'" \
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

We also support multi-turn generation and training (tool use, games, etc.).
Reference example for training to play a Sliding Puzzle Game:
```sh
uv run python examples/run_grpo_sliding_puzzle.py 
```

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

Refer to `examples/configs/sft.yaml` for a full list of parameters that can be overridden.

#### Multi-node

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_sft.py --config examples/configs/sft.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/sft_llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='sft-llama8b'" \
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

If you have access to more GPUs, you can update the experiment accordingly. To run on 8 GPUs, we update the cluster configuration and switch to an 8B Llama3.1 Instruct model:

```sh
uv run python examples/run_dpo.py \
  policy.model_name="meta-llama/Llama-3.1-8B-Instruct" \
  policy.train_global_batch_size=256 \
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

Refer to [dpo.yaml](examples/configs/dpo.yaml) for a full list of parameters that can be overridden. For an in-depth explanation of how to add your own DPO dataset, refer to the [DPO documentation](docs/guides/dpo.md).

#### Multi-node

For distributed DPO training across multiple nodes, modify the following script for your use case:

```sh
# Run from the root of NeMo-Reinforcer repo
## number of nodes to use for your job
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_dpo.py --config examples/configs/dpo.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 dpo.val_global_batch_size=32 checkpointing.checkpoint_dir='results/dpo_llama81_2nodes' logger.wandb_enabled=True logger.wandb.name='dpo-llama1b'" \
RAY_DEDUP_LOGS=0 \
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
