import json
import ray
import os
import random
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy as NASS
from dataclasses import dataclass
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import register_fsdp_forward_method


@dataclass
class WorkerGroupResources:
    num_nodes: int
    num_cpus_per_worker: int = 16  # 128 hyperthread / 8 gpu = 16 cpu/gpu
    num_gpus_per_node: int = 8  # will always be true on slurm


@dataclass
class NodeInfo:
    node_id: str
    node_rank: int
    node_ip: str


# Define the coordinator and worker
class RayClusterCoordinator:
    def __init__(
        self, worker_cls: type["ModelTrainer"], worker_resources: WorkerGroupResources
    ) -> None:
        self.worker_cls = worker_cls
        self.worker_resources = worker_resources
        self.num_workers = (
            worker_resources.num_nodes * worker_resources.num_gpus_per_node
        )
        self.num_workers_per_node = worker_resources.num_gpus_per_node

        ray_available_workers = int(ray.cluster_resources()["worker_units"])
        assert self.num_workers // self.num_workers_per_node <= ray_available_workers, (
            f"Only {ray_available_workers} workers available, which is not enough to schedule {self.num_workers} workers with {self.num_workers_per_node} workers per node"
        )

        self.workers_initialized = False

        worker_node_info, self.num_physical_nodes = self._get_schedulable_worker_info()
        print(f"Worker node info: {worker_node_info=}")
        print(f"Num physical nodes: {self.num_physical_nodes=}")
        # Assume there's one worker per GPU
        self.workers = [
            worker_cls.options(
                num_gpus=1,
                num_cpus=worker_resources.num_cpus_per_worker,
                resources={"worker_units": 1},
                # Use NodeAffinitySchedulingStrategy to ensure each worker is placed on a specific node
                # node_id: Unique ID of the target node for this worker
                # soft=False: Strictly enforce placement on the specified node (no fallback to other nodes)
                scheduling_strategy=NASS(
                    node_id=worker_node_info[i].node_id, soft=False
                ),
            ).remote(
                i,
                self.num_workers,
                worker_node_info[i].node_rank,
                worker_node_info[i].node_ip,  # TODO: probably can delete this
                worker_node_info[
                    0
                ].node_ip,  # Arbitrarily make the first worker's hots the master
                self.num_workers_per_node,
            )
            for i in range(self.num_workers)
        ]

    def _get_schedulable_worker_info(self) -> tuple[list[NodeInfo], int]:
        """Collects information about available worker nodes in the Ray cluster and prepares
        scheduling information for worker actors.

        This method:
        1. Identifies all alive worker nodes with 'worker_units' resources
        2. Sorts them by NodeID for consistent allocation
        3. Calculates how many physical nodes are needed based on workers per node
        4. Verifies that enough nodes are available
        5. Creates a list of NodeInfo objects for each worker

        Returns:
            tuple: (worker_node_info, num_nodes_required)
                - worker_node_info: List of NodeInfo objects containing node_id, node_rank, and node_ip for each worker
                - num_nodes_required: Number of physical nodes needed for all workers

        Raises:
            AssertionError: If there aren't enough nodes available to schedule all workers
        """
        # Get list of alive worker nodes sorted by NodeID for deterministic allocation
        worker_node_info = []
        worker_nodes = sorted(
            [
                node
                for node in ray.nodes()
                if (node["Alive"] and "worker_units" in node["Resources"])
            ],
            key=lambda x: x["NodeID"],
        )

        # Calculate required nodes and verify availability
        num_nodes_required = self.num_workers // self.num_workers_per_node
        num_nodes_available = len(worker_nodes)
        assert num_nodes_required <= num_nodes_available

        # Create worker info entries - one per GPU across all needed nodes
        worker_nodes = worker_nodes[:num_nodes_required]
        for worker_node_id, worker_node in enumerate(worker_nodes):
            for _ in range(self.num_workers_per_node):
                worker_node_info.append(
                    NodeInfo(
                        worker_node["NodeID"], worker_node_id, worker_node["NodeName"]
                    )
                )

        return worker_node_info, num_nodes_required

    def initialize_workers(self, **kwargs):
        self.worker_init_kwargs = kwargs
        ray.get([w.initialize.remote(**kwargs) for i, w in enumerate(self.workers)])
        self.workers_initialized = True

    def run(self, *args, **kwargs):
        if not self.workers_initialized:
            raise ValueError("""Cannot run workers without initializing them first.
                              Please call the initialize_workers method of your cluster coordinator first.""")

        worker_results = ray.get([w.run.remote(*args, **kwargs) for w in self.workers])
        return worker_results


class Worker:
    def __init__(
        self,
        process_id,
        world_size,
        physical_node_id,
        physical_node_ip,
        master_addr: str,
        num_workers_per_node: int,
    ):
        self.process_id = process_id
        self.world_size = world_size
        self.physical_node_id = physical_node_id
        self.host_ip = physical_node_ip
        self.master_addr = master_addr
        self.logical_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
        print(
            f"DEBUG: {self.logical_gpu_id=} {(process_id, world_size, physical_node_id, physical_node_ip, master_addr, num_workers_per_node)=}"
        )
        self.num_workers_per_node = num_workers_per_node

    def get_process_id(self):
        return self.process_id

    def get_host_ip(self):
        return self.host_ip

    def get_logical_gpu_id(self):
        return self.logical_gpu_id

    def get_physical_node_id(self):
        return self.physical_node_id

    def initialize(self):
        # Set distributed training environment variables
        os.environ["RANK"] = str(self.process_id)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.logical_gpu_id)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.num_workers_per_node)
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group("nccl")

    def run(self, *args, **kwargs):
        raise NotImplementedError


@ray.remote
class ModelTrainer(Worker):
    def __init__(
        self,
        process_id,
        world_size,
        physical_node_id,
        physical_node_ip,
        master_addr,
        num_workers_per_node,
    ):
        super().__init__(
            process_id,
            world_size,
            physical_node_id,
            physical_node_ip,
            master_addr,
            num_workers_per_node,
        )

    def run(self, hf_model_name):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        ####local_device = torch.device(f"cuda:{rank}")
        ####torch.cuda.set_device(local_device)

        print(f"[Rank {rank}] Loading model {hf_model_name} on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # TODO: could oom?
        # ------------------------------------------------
        # 3) Move to GPU + Composable FSDP
        #    (Initialize device mesh, shard submodules, then shard entire model)
        # ------------------------------------------------
        model.cuda()

        # Create a device mesh with 'world_size' GPUs in a 1D arrangement.
        mesh = init_device_mesh("cuda", (world_size,))

        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32
        buffer_dtype = torch.float32

        mp = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        model = FullyShardedDataParallel(
            model,
            device_mesh=mesh,
            auto_wrap_policy=size_based_auto_wrap_policy,
            mixed_precision=mp,
        )

        # Optionally register "generate" as the forward method so FSDP can handle it properly.
        register_fsdp_forward_method(model, "generate")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()

        num_epochs = 2
        num_batches = 10
        batch_size = 2
        seq_length = 16
        vocab_size = tokenizer.vocab_size or 32000

        print(f"[Rank {rank}] Starting synthetic training...")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)
        for epoch in range(num_epochs):
            for step in range(num_batches):
                input_ids = torch.ones(
                    (batch_size, seq_length), device="cuda", dtype=torch.long
                ) * (vocab_size - 1)
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = torch.square(outputs.logits.view(-1)[0])
                loss.backward()
                optimizer.step()

                if rank == 0:
                    print(
                        f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.8f}",
                        flush=True,
                    )

        if rank == 0:
            expected_loss = 0.012502908706665039
            print(
                f"[Rank {rank}] Testing loss is close to expected loss: {expected_loss}"
            )
            torch.testing.assert_close(loss.item(), expected_loss)
            print("Yay! Loss was close :)")


if __name__ == "__main__":
    ray.init(address="auto", logging_level=0)
    print(json.dumps(json.loads(os.environ["RAY_JOB_CONFIG_JSON_ENV_VAR"]), indent=4))
    driver_args = json.loads(os.environ["RAY_JOB_CONFIG_JSON_ENV_VAR"])["runtime_env"][
        "driver_args"
    ]

    # TODO: very simple, need to think thru CLI
    trainer_args = driver_args["trainer"]
    trainer_resources = trainer_args.pop("resources")
    worker_resources = WorkerGroupResources(**trainer_resources)

    coordinator = RayClusterCoordinator(ModelTrainer, worker_resources)
    coordinator.initialize_workers()
    print("Initialized workers")
    # Get the job configuration set during launch.
    # This is automatically set by Ray
    coordinator.run(**trainer_args)
    print("Finished")
