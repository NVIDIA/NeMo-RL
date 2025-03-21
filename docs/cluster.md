# Cluster start

- [Cluster start](#cluster-start)
  - [Slurm](#slurm)
    - [Batched Job Submission](#batched-job-submission)
    - [Interactive Launching](#interactive-launching)
  - [Kubernetes](#kubernetes)

## Slurm

:::{tip}
It is important to set `UV_CACHE_DIR` to a directory that can be read from all workers before
running any `uv run` command. This ensures a fast startup time since all workers can re-use the same cache.

```sh
export UV_CACHE_DIR=/path/that/all/workers/can/access/uv_cache
```
:::

### Batched Job Submission

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

COMMAND="bash -c 'uv pip install -e .; uv run ./examples/run_grpo.py'" \
RAY_DEDUP_LOGS=0 \
UV_CACHE_DIR=YOUR_UV_CACHE_DIR \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=$((NUM_ACTOR_NODES + 1)) \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```

Notes:
* Some clusters may or may not need `--gres=gpu:8` to be added to the `sbatch` command.
* Setting `UV_CACHE_DIR` to a shared directory accessible by all worker nodes is critical for performance. Without this, the `uv` package manager will need to synchronize dependencies separately for each worker, which can significantly increase startup times and create unnecessary network traffic.

Which will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Make note of the the job submission number. Once the job begins you can track it's process in the driver logs which you can `tail`:
```sh
tail -f 1980204-logs/ray-driver.log
```

### Interactive Launching
To run interactively, launch the same command as the [Batched Job Submission](#batched-job-submission) except omit the `COMMAND` line:
```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

RAY_DEDUP_LOGS=0 \
UV_CACHE_DIR=YOUR_UV_CACHE_DIR \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=$((NUM_ACTOR_NODES + 1)) \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```
Which will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Once the ray cluster is up, a script should be created to attach to the ray head node,
which you can use launch experiments.
```sh
bash 1980204-attach.sh
```
Now that you are on the head node, you can launch the command like so:
```sh
uv venv -p python3.12.9 .venv
uv pip install -e .
uv run ./examples/run_grpo.py
```

## Kubernetes

TBD
