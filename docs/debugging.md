# Debugging in NeMo RL

This guide explains how to debug NeMo RL applications, covering two scenarios. It first outlines the procedure for debugging distributed Ray worker/actor processes using the Ray Distributed Debugger within a SLURM environment, and then details debugging the main driver script.

## Debugging in the Worker/Actors (on SLURM)

Since Ray programs can spawn many workers/actors, we need to use the Ray Distributed Debugger
to properly jump to the breakpoint on each worker.

### Prerequisites

* Install [Ray Debugger VS Code/Cursor extension](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html).
* Launch the [interactive cluster](./cluster.md#interactive-launching) with `ray.sub`.
* Launch VS Code/Cursor on the SLURM login node (where `squeue`/`sbatch` is available).
* Add `breakpoint()` in your code under actors & tasks (i.e. classes or functions decorated with `@ray.remote`).
* **Ensure** `RAY_DEBUG=legacy` is not set since this debugging requires the default distributed debugger.

### Port-forwarding from the Head Node

From the SLURM login node, query the nodes used by the interactive `ray.sub` job as follows:

```sh
teryk@slurm-login:~$ squeue --me
             JOBID PARTITION        NAME     USER ST       TIME  NODES NODELIST(REASON)
           2504248     batch ray-cluster   terryk  R      15:01      4 node-12,node-[22,30],node-49
```

The first node is always the head node, so we need to port forward the dashboard port to the login node:

```sh
# Traffic from the login node's $LOCAL is forwarded to node-12:$DASHBOARD_PORT
# - If you haven't changed the default DASHBOARD_PORT in ray.sub, it is likely 8265
# - Choose a LOCAL_PORT that isn't taken. If the cluster is multi-tenant, 8265
#   on the login node is likely taken by someone else.
ssh -L $LOCAL_PORT:localhost:$DASHBOARD_PORT -N node-12

# Example chosing a port other than 8265 for the LOCAL_PORT
ssh -L 52640:localhost:8265 -N node-12
```

Example output from the port-forwarding with `ssh` may print logs like this, where the warning is expected:

```text
Warning: Permanently added 'node-12' (ED25519) to the list of known hosts.
bind [::1]:52640: Cannot assign requested address
```

### Open the Ray Debugger Extension

In VS Code/Cursor, open the Ray Debugger extension by clicking on the Ray icon in the activity bar or by searching for "View: Show Ray Debugger" in the command palette (Ctrl+Shift+P or Cmd+Shift+P).

![Ray Debugger Extension Step 1](./assets/ray-debug-step1.png)

### Add the Ray Cluster

Click on the "Add Cluster" button in the Ray Debugger panel.

![Ray Debugger Extension Step 2](./assets/ray-debug-step2.png)

Enter the address and port you set up in the port forwarding step. If you followed the example above using port 52640, you would enter:

![Ray Debugger Extension Step 3](./assets/ray-debug-step3.png)

### Add a Breakpoint and Run Your Program

All breakpoints that are reached while the program is running will be visible in the Ray Debugger Panel dropdown for the cluster `127.0.0.1:52640`. Click
`Start Debugging` to jump to one worker's breakpoint.

Note that you can jump between breakpoints across all workers with this process.

![Ray Debugger Extension Step 4](./assets/ray-debug-step4.png)

## Debugging in the Driver Script

By default, setting breakpoints in the driver script (outside of  `@ray.remote`) will not pause program execution when using Ray. To enable pausing at these breakpoints, set the environment variable to `RAY_DEBUG=legacy`:

```sh
RAY_DEBUG=legacy uv run ....
```
