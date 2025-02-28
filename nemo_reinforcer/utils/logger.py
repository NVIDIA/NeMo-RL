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


import os
import re
import glob
from abc import ABC, abstractmethod
import logging
from typing import List, Any, Dict, Optional, TypedDict
import wandb
from rich.console import Console
from rich.panel import Panel
from rich.box import ROUNDED
from rich.logging import RichHandler

from nemo_reinforcer.data.interfaces import LLMMessageLogType
from torch.utils.tensorboard import SummaryWriter

# Flag to track if rich logging has been configured
_rich_logging_configured = False


class WandbConfig(TypedDict):
    project: str
    name: str


class TensorboardConfig(TypedDict):
    log_dir: str


class LoggerConfig(TypedDict):
    log_dir: str
    wandb_enabled: bool
    tensorboard_enabled: bool
    wandb: WandbConfig
    tensorboard: TensorboardConfig


class LoggerInterface(ABC):
    """Abstract base class for logger backends."""

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, Any], step: int, prefix: Optional[str] = ""
    ) -> None:
        """Log a dictionary of metrics."""
        pass

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log dictionary of hyperparameters."""
        pass


class TensorboardLogger(LoggerInterface):
    """Tensorboard logger backend."""

    def __init__(self, cfg: TensorboardConfig, log_dir: Optional[str] = None):
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Initialized TensorboardLogger at {log_dir}")

    def log_metrics(
        self, metrics: Dict[str, Any], step: int, prefix: Optional[str] = ""
    ) -> None:
        """Log metrics to Tensorboard.

        Args:
            metrics: Dict of metrics to log
            step: Global step value
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            if prefix:
                name = f"{prefix}/{name}"
            self.writer.add_scalar(name, value, step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to Tensorboard.

        Args:
            params: Dictionary of hyperparameters to log
        """
        # Flatten the params because add_hparams does not support nested dicts
        self.writer.add_hparams(flatten_dict(params), {})


class WandbLogger(LoggerInterface):
    """Weights & Biases logger backend."""

    def __init__(self, cfg: WandbConfig, log_dir: Optional[str] = None):
        self.run = wandb.init(**cfg, dir=log_dir)
        print(
            f"Initialized WandbLogger for project {cfg.get('project')}, run {cfg.get('name')} at {log_dir}"
        )

    def log_metrics(
        self, metrics: Dict[str, Any], step: int, prefix: Optional[str] = ""
    ) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dict of metrics to log
            step: Global step value
            prefix: Optional prefix for metric names
        """
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        self.run.log(metrics, step=step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to wandb.

        Args:
            params: Dict of hyperparameters to log
        """
        self.run.config.update(params)


class Logger(LoggerInterface):
    """Main logger class that delegates to multiple backend loggers."""

    def __init__(self, cfg: LoggerConfig):
        """Initialize the logger.

        Args:
            cfg: Config dict with the following keys:
                - wandb_enabled
                - tensorboard_enabled
                - wandb
                - tensorboard
        """
        self.loggers = []

        self.base_log_dir = cfg["log_dir"]
        os.makedirs(self.base_log_dir, exist_ok=True)

        if cfg["wandb_enabled"]:
            wandb_log_dir = os.path.join(self.base_log_dir, "wandb")
            os.makedirs(wandb_log_dir, exist_ok=True)
            wandb_logger = WandbLogger(cfg["wandb"], log_dir=wandb_log_dir)
            self.loggers.append(wandb_logger)

        if cfg["tensorboard_enabled"]:
            tensorboard_log_dir = os.path.join(self.base_log_dir, "tensorboard")
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            tensorboard_logger = TensorboardLogger(
                cfg["tensorboard"], log_dir=tensorboard_log_dir
            )
            self.loggers.append(tensorboard_logger)

        if not self.loggers:
            print("No loggers initialized")

    def log_metrics(
        self, metrics: Dict[str, Any], step: int, prefix: Optional[str] = ""
    ) -> None:
        """Log metrics to all enabled backends.

        Args:
            metrics: Dict of metrics to log
            step: Global step value
            prefix: Optional prefix for metric names
        """
        for logger in self.loggers:
            logger.log_metrics(metrics, step, prefix)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all enabled backends.

        Args:
            params: Dict of hyperparameters to log
        """
        for logger in self.loggers:
            logger.log_hyperparams(params)


def flatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    result = {}

    def _flatten(d, parent_key=""):
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                _flatten(value, new_key)
            else:
                result[new_key] = value

    _flatten(d)
    return result


"""
Rich Console Logging Functionality
---------------------------------
Functions for setting up rich console logging and visualizing model outputs.
"""


def configure_rich_logging(
    level: str = "INFO", show_time: bool = True, show_path: bool = True
) -> None:
    """Configure rich logging for more visually appealing log output.

    Args:
        level: The logging level to use
        show_time: Whether to show timestamps in logs
        show_path: Whether to show file paths in logs
    """
    global _rich_logging_configured

    # Only configure if not already done
    if not _rich_logging_configured:
        # Configure logging with rich handler
        logging.basicConfig(
            level=level.upper(),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    rich_tracebacks=True,
                    show_time=show_time,
                    show_path=show_path,
                    markup=True,
                )
            ],
        )
        _rich_logging_configured = True


def print_message_log_samples(
    message_logs: List[LLMMessageLogType],
    rewards: List[float],
    num_samples: int = 5,
    step: int = 0,
) -> None:
    """Visualization for message logs and rewards using a more visual approach with emoji indicators and horizontal layout.

    Args:
        message_logs: List of message logs to sample from
        rewards: List of rewards corresponding to each message log
        num_samples: Number of samples to display (default: 5)
        step: Current training step (for display purposes)
    """
    # Make sure rich logging is configured before printing
    configure_rich_logging(level="INFO")

    if not message_logs or not rewards:
        return

    if num_samples <= 0:
        return

    # Sample up to num_samples (or all if less)
    num_to_show = min(num_samples, len(message_logs))
    indices = list(range(len(message_logs)))

    # If we have more samples than needed, prioritize showing a mix of high and low rewards
    if len(indices) > num_to_show:
        # Sort indices by reward
        sorted_indices = sorted(indices, key=lambda i: rewards[i], reverse=True)
        # Take some from the top and some from the bottom
        half = num_to_show // 2
        indices = sorted_indices[:half] + sorted_indices[-half:]
        # If num_to_show is odd, add a middle sample
        if num_to_show % 2 == 1:
            middle_idx = len(sorted_indices) // 2
            indices.append(sorted_indices[middle_idx])
        indices = indices[:num_to_show]

    console = Console()

    # Header with step information
    console.rule(f"[bold bright_white on purple4]TRAINING STEP {step}")

    # Count the unique reward values
    all_rewards = rewards.copy()
    unique_rewards = sorted(set(all_rewards))
    reward_counts = {r: all_rewards.count(r) for r in unique_rewards}

    # Create a bar chart for discrete reward levels
    max_count = max(reward_counts.values()) if reward_counts else 1

    # Create discrete reward level visualization
    discrete_lines = []
    discrete_lines.append("[bold bright_white]Discrete Reward Levels:[/]")

    # Get emoji for each reward level
    def get_reward_emoji(reward):
        if reward >= 0.7:
            return "🔥"  # Excellent
        elif reward >= 0.3:
            return "✨"  # Good
        elif reward >= -0.5:
            return "🟠"  # Poor
        else:
            return "🔴"  # Very poor

    # Create a bar for each discrete reward level
    for reward in unique_rewards:
        count = reward_counts[reward]
        emoji = get_reward_emoji(reward)
        bar_len = int((count / max_count) * 20)

        # Choose different bar characters and colors
        if reward > 0.5:
            bar_char = "█"
            color = "bright_green"
        elif reward > 0:
            bar_char = "█"
            color = "green"
        elif reward == 0:
            bar_char = "▒"
            color = "bright_white"
        elif reward > -0.5:
            bar_char = "▓"
            color = "orange3"
        else:
            bar_char = "█"
            color = "red"

        bar = f"[{color}]{bar_char * bar_len}[/]"
        # Format with color based on reward value
        discrete_lines.append(
            f"{emoji} Reward [bold {color}]{reward:.4f}[/]: {bar} ({count} samples)"
        )

    # Create a summary panel
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    stats_text = (
        f"[bold]Batch Summary[/]\n"
        f"Total Samples: [bright_yellow]{len(all_rewards)}[/]\n"
        f"Avg Reward: [bright_blue]{avg_reward:.4f}[/]\n"
        f"Min: [orange3]{min(all_rewards):.4f}[/] | Max: [bright_green]{max(all_rewards):.4f}[/]\n\n"
        + "\n".join(discrete_lines)
    )

    stats_panel = Panel(
        stats_text,
        title="[bold purple4]Reward Statistics",
        border_style="purple4",
        box=ROUNDED,
    )

    # Display the stats panel
    console.print(stats_panel)

    # Display the samples with horizontal layout
    console.print("\n[bold bright_white]Sample Conversations[/]")

    # Helper function to safely render content that might have problematic markups
    def safe_render(content, role_color):
        # Fix common problematic patterns that might break Rich markup
        # Replace any standalone [/ without matching closing bracket
        content = content.replace("[/", "\\[/")
        # Replace any standalone [ that isn't followed by a valid tag with escaped version
        import re

        content = re.sub(r"\[(?![a-z_]+\s|/[a-z_]+\])", "\\[", content)
        return f"[{role_color}]{content}[/]"

    for i, idx in enumerate(indices):
        message_log = message_logs[idx]
        reward = rewards[idx]

        # Format each message in the conversation
        message_parts = []
        for msg in message_log:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            # Choose color based on role - using muted, elegant colors
            if role == "SYSTEM":
                message_parts.append(
                    f"[bold #8A2BE2]{role}:[/] {safe_render(content, '#8A2BE2')}"
                )
            elif role == "USER":
                message_parts.append(
                    f"[bold #4682B4]{role}:[/] {safe_render(content, '#4682B4')}"
                )
            elif role == "ASSISTANT":
                message_parts.append(
                    f"[bold #2E8B57]{role}:[/] {safe_render(content, '#2E8B57')}"
                )
            else:
                message_parts.append(f"[bold]{role}:[/] {content}")

        # Get reward emoji
        emoji = get_reward_emoji(reward)

        # Choose color based on reward
        if reward > 0.5:
            color = "bright_green"
        elif reward > 0:
            color = "green"
        elif reward == 0:
            color = "bright_white"
        elif reward > -0.5:
            color = "orange3"
        else:
            color = "red"

        content = "\n\n".join(message_parts)

        panel = Panel(
            content,
            title=f"[bold]{emoji} Sample {i + 1} | Reward: {reward:.4f}",
            border_style=color,
            box=ROUNDED,
        )

        console.print(panel)
        console.print("")  # Add some spacing

    console.rule("[bold bright_white on purple4]End of Samples")


def get_next_experiment_dir(base_log_dir):
    """Create a new experiment directory with an incremented ID.

    Args:
        base_log_dir (str): The base log directory path

    Returns:
        str: Path to the new experiment directory with incremented ID
    """
    # Check if the log directory already contains an experiment ID pattern (e.g., /exp_001/)
    pattern = re.compile(r"exp_(\d+)")
    next_exp_id = 1

    # Check for existing experiment directories
    existing_dirs = glob.glob(os.path.join(base_log_dir, "exp_*"))

    if existing_dirs:
        # Extract experiment IDs and find the maximum
        exp_ids = []
        for dir_path in existing_dirs:
            match = pattern.search(dir_path)
            if match:
                exp_ids.append(int(match.group(1)))

        if exp_ids:
            # Increment the highest experiment ID
            next_exp_id = max(exp_ids) + 1

    # Format the new log directory with the incremented experiment ID
    new_log_dir = os.path.join(base_log_dir, f"exp_{next_exp_id:03d}")

    # Create the new log directory
    os.makedirs(new_log_dir, exist_ok=True)

    return new_log_dir
