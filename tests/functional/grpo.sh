#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)

set -eou pipefail

LOG_DIR=$SCRIPT_DIR/$(basename $0 .sh)-logs
JSON_METRICS=$LOG_DIR/$(basename $0 .sh).json
RUN_LOG=$LOG_DIR/$(basename $0 .sh).log
export RAY_DEDUP_LOGS=0
export UV_CACHE_DIR=$PROJECT_ROOT/uv_cache

mkdir -p $LOG_DIR

cd $PROJECT_ROOT
uv run $PROJECT_ROOT/examples/run_grpo_math.py \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=10 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

cd $SCRIPT_DIR
uv run json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run check_metrics.py $JSON_METRICS \
    'data["timing/train/policy_refit"]["10"] < 3.0' \
    'data["timing/train/total_step_time"]["10"] < 20.0' \
    'data["timing/validation/generation"]["10"] < 3.0' \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'data["validation/avg_length"]["10"] < 1024' \

