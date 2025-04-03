# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json

from nemo_reinforcer.distributed.virtual_cluster import init_ray, RayVirtualCluster
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Torch DCP checkpoint to HF checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json file in the checkpoint directory",
    )
    parser.add_argument(
        "--dcp-ckpt-path", type=str, default=None, help="Path to DCP checkpoint"
    )
    parser.add_argument(
        "--hf-ckpt-path", type=str, default=None, help="Path to save HF checkpoint"
    )
    # Parse known args for the script
    args = parser.parse_args()

    return args


def main():
    """Main entry point."""
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    dcp_ckpt = args.dcp_ckpt_path
    hf_ckpt = args.hf_ckpt_path

    # Extract individual configs for easier access
    policy_config = config["policy"]
    cluster_config = config["cluster"]

    init_ray()

    cluster = RayVirtualCluster(
        name="convert_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )

    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        weights_path=dcp_ckpt,
        init_optimizer=False,
    )

    policy.save_checkpoint(
        weights_path=os.path.abspath(hf_ckpt),
        save_hf=True,
        save_torch_dist=False,
    )

    print(f"Saved HF checkpoint to: {hf_ckpt}-hf")

    cluster.shutdown()
    policy.worker_group.shutdown()


if __name__ == "__main__":
    main()
