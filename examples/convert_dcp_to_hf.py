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
from omegaconf import OmegaConf

from nemo_reinforcer.distributed.virtual_cluster import RayVirtualCluster
from nemo_reinforcer.models.policy.hf_policy import HfPolicy
from nemo_reinforcer.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Torch DCP checkpoint to HF checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--dcp-ckpt-path", type=str, default=None, help="Path to DCP checkpoint"
    )
    parser.add_argument(
        "--hf-ckpt-path", type=str, default=None, help="Path to save HF checkpoint"
    )
    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "sft.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = OmegaConf.merge(config, overrides)

    dcp_ckpt = args.dcp_ckpt_path
    hf_ckpt = args.hf_ckpt_path

    # Extract individual configs for easier access
    policy_config = config["policy"]
    cluster_config = config["cluster"]

    cluster = RayVirtualCluster(
        name="sft_cluster",
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
        weights_path=hf_ckpt,
        save_hf=True,
        save_torch_dist=False,
    )
    print(f"Saved HF checkpoint to: {hf_ckpt}-hf")


if __name__ == "__main__":
    main()
