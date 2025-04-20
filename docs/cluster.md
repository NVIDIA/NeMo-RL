# Cluster start

- [Cluster start](#cluster-start)
  - [Slurm](#slurm)
    - [Batched Job Submission](#batched-job-submission)
    - [Interactive Launching](#interactive-launching)
  - [Kubernetes](#kubernetes)

## Slurm

### Batched Job Submission

Launch the following lines from your experiment directory. All ray logs will be stored in your experiment directory.

```sh
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)
WD=$(readlink -f $PWD) # Working directory with all soft links expanded
REINFORCER_DIR=YOUR_ABSOLUTE_REINFORCER_DIR

COMMAND="cd $REINFORCER_DIR; uv pip install -e .; uv run ./examples/run_grpo_math.py" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$WD:$WD,$REINFORCER_DIR:$REINFORCER_DIR" \
HF_TOKEN=YOUR_HF_TOKEN \
WANDB_API_KEY=YOUR_WANDB_API_KEY \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    $REINFORCER_DIR/ray.sub
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

:::{note}
`UV_CACHE_DIR` defaults to `$SLURM_SUBMIT_DIR/uv_cache` and is mounted to head and worker nodes
to ensure fast `venv` creation. 

If you would like to override it to somewhere else all head/worker nodes can access, you may set it
via:

```sh
...
UV_CACHE_DIR=/path/that/all/workers/and/head/can/access \
sbatch ... \
    ray.sub
```
:::

### Interactive Launching

:::{tip}
A key advantage of running interactively on the head node is the ability to execute multiple multi-node jobs without needing to requeue in the SLURM job queue. This means during debugging sessions, you can avoid submitting a new `sbatch` command each time and instead debug and re-submit your Reinforcer job directly from the interactive session.
:::

To run interactively, launch the same command as the [Batched Job Submission](#batched-job-submission) except omitting the `COMMAND` line:

```sh
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)
WD=$(readlink -f $PWD) # Working directory with all soft links expanded
REINFORCER_DIR=YOUR_ABSOLUTE_REINFORCER_DIR

CONTAINER=YOUR_CONTAINER \
MOUNTS="$WD:$WD,$REINFORCER_DIR:$REINFORCER_DIR" \
HF_TOKEN=YOUR_HF_TOKEN \
WANDB_API_KEY=YOUR_WANDB_API_KEY \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    $REINFORCER_DIR/ray.sub
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
cd $REINFORCER_DIR
uv venv .venv
uv pip install -e .
uv run ./examples/run_grpo_math.py
```

## Kubernetes

TBD
