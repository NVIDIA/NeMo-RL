#!/usr/bin/env python3

import argparse
import os
import pprint

from omegaconf import OmegaConf

from nemo_reinforcer.algorithms.sft import MasterConfig, sft_train, setup
from nemo_reinforcer.distributed.virtual_cluster import init_ray
from nemo_reinforcer.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SFT training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
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

    config = OmegaConf.load(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        override_conf = OmegaConf.from_cli()
        print(f"Overrides: {override_conf}")
        config = OmegaConf.merge(config, override_conf)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")

    init_ray()
    (
        policy,
        cluster,
        dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        sft_task_spec,
    ) = setup(config)
    sft_train(
        policy,
        dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        sft_task_spec,
    )


if __name__ == "__main__":
    main()
