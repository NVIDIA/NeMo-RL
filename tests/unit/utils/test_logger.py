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

import shutil
import tempfile
from unittest.mock import patch

import pytest

from nemo_reinforcer.utils.logger import (
    Logger,
    TensorboardLogger,
    WandbLogger,
    flatten_dict,
)


class TestFlattenDict:
    """Test the flatten_dict utility function."""

    def test_empty_dict(self):
        """Test flattening an empty dictionary."""
        assert flatten_dict({}) == {}

    def test_flat_dict(self):
        """Test flattening a dictionary that is already flat."""
        d = {"a": 1, "b": 2, "c": 3}
        assert flatten_dict(d) == d

    def test_nested_dict(self):
        """Test flattening a nested dictionary."""
        d = {"a": 1, "b": {"c": 2, "d": 3}, "e": {"f": {"g": 4}}}
        expected = {"a": 1, "b.c": 2, "b.d": 3, "e.f.g": 4}
        assert flatten_dict(d) == expected

    def test_custom_separator(self):
        """Test flattening with a custom separator."""
        d = {"a": 1, "b": {"c": 2, "d": 3}}
        expected = {"a": 1, "b_c": 2, "b_d": 3}
        assert flatten_dict(d, sep="_") == expected


class TestTensorboardLogger:
    """Test the TensorboardLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("nemo_reinforcer.utils.logger.SummaryWriter")
    def test_init(self, mock_summary_writer, temp_dir):
        """Test initialization of TensorboardLogger."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        # The log_dir is passed to SummaryWriter but not stored as an attribute
        mock_summary_writer.assert_called_once_with(log_dir=temp_dir)

    @patch("nemo_reinforcer.utils.logger.SummaryWriter")
    def test_log_metrics(self, mock_summary_writer, temp_dir):
        """Test logging metrics to TensorboardLogger."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that add_scalar was called for each metric
        mock_writer = mock_summary_writer.return_value
        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("loss", 0.5, 10)
        mock_writer.add_scalar.assert_any_call("accuracy", 0.8, 10)

    @patch("nemo_reinforcer.utils.logger.SummaryWriter")
    def test_log_metrics_with_prefix(self, mock_summary_writer, temp_dir):
        """Test logging metrics with a prefix to TensorboardLogger."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        prefix = "train"
        logger.log_metrics(metrics, step, prefix)

        # Check that add_scalar was called for each metric with prefix
        mock_writer = mock_summary_writer.return_value
        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("train/loss", 0.5, 10)
        mock_writer.add_scalar.assert_any_call("train/accuracy", 0.8, 10)

    @patch("nemo_reinforcer.utils.logger.SummaryWriter")
    def test_log_hyperparams(self, mock_summary_writer, temp_dir):
        """Test logging hyperparameters to TensorboardLogger."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        params = {"lr": 0.001, "batch_size": 32, "model": {"hidden_size": 128}}
        logger.log_hyperparams(params)

        # Check that add_hparams was called with flattened params
        mock_writer = mock_summary_writer.return_value
        mock_writer.add_hparams.assert_called_once()
        # First argument should be flattened dict
        called_params = mock_writer.add_hparams.call_args[0][0]
        assert called_params == {
            "lr": 0.001,
            "batch_size": 32,
            "model.hidden_size": 128,
        }


class TestWandbLogger:
    """Test the WandbLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("nemo_reinforcer.utils.logger.wandb")
    def test_init_custom_config(self, mock_wandb, temp_dir):
        """Test initialization of WandbLogger with custom config."""
        cfg = {
            "project": "custom-project",
            "name": "custom-run",
            "entity": "custom-entity",
            "group": "custom-group",
            "tags": ["tag1", "tag2"],
        }
        WandbLogger(cfg, log_dir=temp_dir)

        mock_wandb.init.assert_called_once_with(
            project="custom-project",
            name="custom-run",
            entity="custom-entity",
            group="custom-group",
            tags=["tag1", "tag2"],
            dir=temp_dir,
        )

    @patch("nemo_reinforcer.utils.logger.wandb")
    def test_log_metrics(self, mock_wandb):
        """Test logging metrics to WandbLogger."""
        cfg = {}
        logger = WandbLogger(cfg)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that log was called with metrics and step
        mock_run = mock_wandb.init.return_value
        mock_run.log.assert_called_once_with(metrics, step=step)

    @patch("nemo_reinforcer.utils.logger.wandb")
    def test_log_metrics_with_prefix(self, mock_wandb):
        """Test logging metrics with a prefix to WandbLogger."""
        cfg = {}
        logger = WandbLogger(cfg)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        prefix = "train"
        logger.log_metrics(metrics, step, prefix)

        # Check that log was called with prefixed metrics and step
        mock_run = mock_wandb.init.return_value
        expected_metrics = {"train/loss": 0.5, "train/accuracy": 0.8}
        mock_run.log.assert_called_once_with(expected_metrics, step=step)

    @patch("nemo_reinforcer.utils.logger.wandb")
    def test_log_hyperparams(self, mock_wandb):
        """Test logging hyperparameters to WandbLogger."""
        cfg = {}
        logger = WandbLogger(cfg)

        params = {"lr": 0.001, "batch_size": 32, "model": {"hidden_size": 128}}
        logger.log_hyperparams(params)

        # Check that config.update was called with params
        mock_run = mock_wandb.init.return_value
        mock_run.config.update.assert_called_once_with(params)


class TestLogger:
    """Test the main Logger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("nemo_reinforcer.utils.logger.WandbLogger")
    @patch("nemo_reinforcer.utils.logger.TensorboardLogger")
    def test_init_no_loggers(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with no loggers enabled."""
        cfg = {
            "wandb_enabled": False,
            "tensorboard_enabled": False,
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 0
        mock_tb_logger.assert_not_called()
        mock_wandb_logger.assert_not_called()

    @patch("nemo_reinforcer.utils.logger.WandbLogger")
    @patch("nemo_reinforcer.utils.logger.TensorboardLogger")
    def test_init_wandb_only(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with only WandbLogger enabled."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": False,
            "wandb": {"project": "test-project"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 1
        mock_wandb_logger.assert_called_once()
        wandb_cfg = mock_wandb_logger.call_args[0][0]
        assert wandb_cfg == {"project": "test-project"}
        mock_tb_logger.assert_not_called()

    @patch("nemo_reinforcer.utils.logger.WandbLogger")
    @patch("nemo_reinforcer.utils.logger.TensorboardLogger")
    def test_init_tensorboard_only(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with only TensorboardLogger enabled."""
        cfg = {
            "wandb_enabled": False,
            "tensorboard_enabled": True,
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 1
        mock_tb_logger.assert_called_once()
        tb_cfg = mock_tb_logger.call_args[0][0]
        assert tb_cfg == {"log_dir": "test_logs"}
        mock_wandb_logger.assert_not_called()

    @patch("nemo_reinforcer.utils.logger.WandbLogger")
    @patch("nemo_reinforcer.utils.logger.TensorboardLogger")
    def test_init_both_loggers(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with both loggers enabled."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 2
        mock_wandb_logger.assert_called_once()
        wandb_cfg = mock_wandb_logger.call_args[0][0]
        assert wandb_cfg == {"project": "test-project"}

        mock_tb_logger.assert_called_once()
        tb_cfg = mock_tb_logger.call_args[0][0]
        assert tb_cfg == {"log_dir": "test_logs"}

    @patch("nemo_reinforcer.utils.logger.WandbLogger")
    @patch("nemo_reinforcer.utils.logger.TensorboardLogger")
    def test_log_metrics(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test logging metrics to all enabled loggers."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Create mock logger instances
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_tb_instance = mock_tb_logger.return_value

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that log_metrics was called on both loggers
        mock_wandb_instance.log_metrics.assert_called_once_with(metrics, step, "")
        mock_tb_instance.log_metrics.assert_called_once_with(metrics, step, "")

    @patch("nemo_reinforcer.utils.logger.WandbLogger")
    @patch("nemo_reinforcer.utils.logger.TensorboardLogger")
    def test_log_hyperparams(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test logging hyperparameters to all enabled loggers."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Create mock logger instances
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_tb_instance = mock_tb_logger.return_value

        params = {"lr": 0.001, "batch_size": 32}
        logger.log_hyperparams(params)

        # Check that log_hyperparams was called on both loggers
        mock_wandb_instance.log_hyperparams.assert_called_once_with(params)
        mock_tb_instance.log_hyperparams.assert_called_once_with(params)
