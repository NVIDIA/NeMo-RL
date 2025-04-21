# Cluster start

- [Cluster start](#cluster-start)
  - [Slurm](#slurm)
    - [Batched Job Submission](#batched-job-submission)
    - [Interactive Launching](#interactive-launching)
    - [Slurm UV\_CACHE\_DIR](#slurm-uv_cache_dir)
  - [Kubernetes](#kubernetes)

## Slurm

### Batched Job Submission

```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

COMMAND="uv run ./examples/run_grpo_math.py" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```

Notes:
* Some clusters may or may not need `--gres=gpu:8` to be added to the `sbatch` command.

Which will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Make note of the the job submission number. Once the job begins you can track it's process in the driver logs which you can `tail`:
```sh
tail -f 1980204-logs/ray-driver.log
```

### Interactive Launching

:::{tip}
A key advantage of running interactively on the head node is the ability to execute multiple multi-node jobs without needing to requeue in the SLURM job queue. This means during debugging sessions, you can avoid submitting a new `sbatch` command each time and instead debug and re-submit your Reinforcer job directly from the interactive session.
:::

To run interactively, launch the same command as the [Batched Job Submission](#batched-job-submission) except omit the `COMMAND` line:
```sh
# Run from the root of NeMo-Reinforcer repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
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
uv run ./examples/run_grpo_math.py
```

### Slurm UV_CACHE_DIR

There several choices for `UV_CACHE_DIR` when using `ray.sub`:

1. (default) `UV_CACHE_DIR` defaults to `$SLURM_SUBMIT_DIR/uv_cache` when not specified the shell environment, and is mounted to head and worker nodes to serve as a persistent cache between runs.
2. Use the warm uv cache from our docker images
    ```sh
    ...
    UV_CACHE_DIR=/home/ray/.cache/uv \
    sbatch ... \
        ray.sub
    ```

(1) is more efficient in general since the cache is not ephemeral and is persisted run to run; but for users that
don't want to persist the cache, you can use (2), which is just as performant as (1) if the `uv.lock` is 
covered by warmed cache.


## Kubernetes

TBD
