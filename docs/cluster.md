# Set Up Clusters

This guide explains how to run NeMo RL with Ray on Slurm or Kubernetes.

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

```{tip}
Some Slurm clusters may or may not need `--gres=gpu:8` to be added to the `sbatch` command.
```

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

### Slurm Environment Variables

All Slurm environment variables described below can be added to the `sbatch`
invocation of `ray.sub`. For example, `GPUS_PER_NODE=8` can be specified as follows:

```sh
GPUS_PER_NODE=8 \
... \
sbatch ray.sub \
   ...
```
#### Common Environment Configuration
``````{list-table}
:header-rows: 1

* - Environment Variable
  - Explanation
* - `CONTAINER`
  - (Required) Specifies the container image to be used for the Ray cluster.
    Use either a docker image from a registry or a squashfs (if using enroot/pyxis).
* - `MOUNTS`
  - (Required) Defines paths to mount into the container. Examples:
    ```md
    * `MOUNTS="$PWD:$PWD"` (mount in current working directory (CWD))
    * `MOUNTS="$PWD:$PWD,/nfs:/nfs:ro"` (mounts the current working directory and `/nfs`, with `/nfs` mounted as read-only)
    ```
* - `COMMAND`
  - Command to execute after the Ray cluster starts. If empty, the cluster idles and enters interactive mode (see the [Slurm interactive instructions](#interactive-launching)).
* - `HF_HOME`
  - Sets the cache directory for huggingface-hub assets (e.g., models/tokenizers).
* - `WANDB_API_KEY`
  - Setting this allows you to use the wandb logger without having to run `wandb login`.
* - `HF_TOKEN`
  - Setting the token used by huggingface-hub. Avoids having to run the `huggingface-cli login`
* - `HF_DATASETS_CACHE`
  - Sets the cache dir for downloaded Huggingface datasets.
``````

:::{tip}
When `HF_TOKEN`, `WANDB_API_KEY`, `HF_HOME`, and `HF_DATASETS_CACHE` are set in your shell environment using `export`, they are automatically passed to `ray.sub`. For instance, if you set:

```sh
export HF_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
this token will be available to your NeMo RL run. Consider adding these exports to your shell configuration file, such as `~/.bashrc`.
:::

#### Advanced Environment Configuration
``````{list-table}
:header-rows: 1

* - Environment Variable
    (and default)
  - Explanation
* - `CPUS_PER_WORKER=128`
  - CPUs each Ray worker node claims. Default is `16 * GPUS_PER_NODE`.
* - `GPUS_PER_NODE=8`
  - Number of GPUs each Ray worker node claims. To determine this, run `nvidia-smi` on a worker node.
* - `BASE_LOG_DIR=$SLURM_SUBMIT_DIR`
  - Base directory for storing Ray logs. Defaults to the Slurm submission directory ([SLURM_SUBMIT_DIR](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_SUBMIT_DIR)).
* - `NODE_MANAGER_PORT=53001`
  - Port for the Ray node manager on worker nodes.
* - `OBJECT_MANAGER_PORT=53003`
  - Port for the Ray object manager on worker nodes.
* - `RUNTIME_ENV_AGENT_PORT=53005`
  - Port for the Ray runtime environment agent on worker nodes.
* - `DASHBOARD_AGENT_GRPC_PORT=53007`
  - gRPC port for the Ray dashboard agent on worker nodes.
* - `METRICS_EXPORT_PORT=53009`
  - Port for exporting metrics from worker nodes.
* - `PORT=6379`
  - Main port for the Ray head node.
* - `RAY_CLIENT_SERVER_PORT=10001`
  - Port for the Ray client server on the head node.
* - `DASHBOARD_GRPC_PORT=52367`
  - gRPC port for the Ray dashboard on the head node.
* - `DASHBOARD_PORT=8265`
  - Port for the Ray dashboard UI on the head node. This is also the port
    used by the Ray distributed debugger.
* - `DASHBOARD_AGENT_LISTEN_PORT=52365`
  - Listening port for the dashboard agent on the head node.
* - `MIN_WORKER_PORT=54001`
  - Minimum port in the range for Ray worker processes.
* - `MAX_WORKER_PORT=54257`
  - Maximum port in the range for Ray worker processes.
``````

:::{note}
For the most part, you will not need to change ports unless these
are already taken by some other service backgrounded on your cluster.
:::

## Kubernetes

TBD