import time
from ray.job_submission import JobSubmissionClient, JobStatus


def main() -> None:
    client = JobSubmissionClient("http://127.0.0.1:8265")
    print("Connected to head!", flush=True)

    # HACK: for now just
    def num_ray_nodes_available() -> int:
        import ray

        ray.init()
        num_gpus_per_node = 8  # hard coded
        num_nodes_avail = (
            int(ray.cluster_resources()["worker_units"]) // num_gpus_per_node
        )
        ray.shutdown()
        return num_nodes_avail

    job_id = client.submit_job(
        entrypoint="RAY_DEDUP_LOGS=0 python3 tests/functional/hello_world_fsdp_llama/train.py",
        runtime_env={
            # TODO: disabling for now since it causes issues if my hf_home is in my working dir and ray
            # wants to upload it to all workers. you get an error like this:
            #   2025-03-02 11:16:48,187 WARNING packaging.py:417 -- File /workspace/hf_home/hub/models--meta-llama--Meta-Llama-3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/model-00001-of-00004.safetensors is very large (4746.15MiB). Consider adding this file to the 'excludes' list to skip uploading it: `ray.init(..., runtime_env={'excludes': ['/workspace/hf_home/hub/models--meta-llama--Meta-Llama-3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/model-00001-of-00004.safetensors']})`
            # "working_dir": "./",
            "driver_args": {
                # Scope each "workergroup"
                "trainer": {
                    "resources": {
                        # TODO: read this in from cli args eventually, but for now just use all available
                        "num_nodes": num_ray_nodes_available(),
                        "num_gpus_per_node": 8,
                        "num_cpus_per_worker": 16,
                    },
                    "hf_model_name": "meta-llama/Llama-3.2-1B",
                }
            },
            "env_vars": {
                # TODO: hardcoded, parametrize
                "HF_HOME": "/workspace/hf_home",
            },
        },
    )

    print(f"Launched job: {job_id}", flush=True)
    prev_logs = ""
    while True:
        status = client.get_job_status(job_id)
        if status in {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}:
            if status in {JobStatus.STOPPED, JobStatus.FAILED}:
                logs = client.get_job_logs(job_id)
                print(logs, flush=True)
            break
        time.sleep(5)
        if status == JobStatus.RUNNING:
            logs = client.get_job_logs(job_id)
            print(logs[len(prev_logs) :], flush=True)
            prev_logs = logs


if __name__ == "__main__":
    main()
