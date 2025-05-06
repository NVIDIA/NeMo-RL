# Set Up Clusters

This guide explains how to initialize NeMo RL clusters.

## Slurm (Batched and Interactive)

 The following code provides instructions on how to use Slurm to run batched job submissions and run jobs interactively.

### Batched Job Submission

```sh
# Run from the root of NeMo RL repo
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
Make note of the the job submission number. Once the job begins, you can track its process in the driver logs which you can `tail`:
```sh
tail -f 1980204-logs/ray-driver.log
```

### Interactive Launching

:::{tip}
A key advantage of running interactively on the head node is the ability to execute multiple multi-node jobs without needing to requeue in the Slurm job queue. This means that during debugging sessions, you can avoid submitting a new `sbatch` command each time. Instead, you can debug and re-submit your NeMo RL job directly from the interactive session.
:::

To run interactively, launch the same command as [Batched Job Submission](#batched-job-submission), but omit the `COMMAND` line:
```sh
# Run from the root of NeMo RL repo
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
Once the Ray cluster is up, a script should be created to attach to the Ray head node,
which you can use to launch experiments.
```sh
bash 1980204-attach.sh
```
Now that you are on the head node, you can launch the command as follows:
```sh
uv run ./examples/run_grpo_math.py
```

### Slurm UV_CACHE_DIR

There several choices for `UV_CACHE_DIR` when using `ray.sub`:

1. (default) `UV_CACHE_DIR` defaults to `$SLURM_SUBMIT_DIR/uv_cache` when not specified the shell environment, and is mounted to head and worker nodes to serve as a persistent cache between runs.
2. Use the warm uv cache from our docker images:
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