#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

LOG_DIR=$SCRIPT_DIR/$(basename $0 .sh)-logs
JSON_METRICS=$LOG_DIR/$(basename $0 .sh).json
RUN_LOG=$LOG_DIR/$(basename $0 .sh).log
export RAY_DEDUP_LOGS=0
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PROJECT_ROOT/uv_cache}
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $LOG_DIR
mkdir -p $LOG_DIR

cd $PROJECT_ROOT
python -u $PROJECT_ROOT/examples/run_dpo.py \
    cluster.gpus_per_node=2 \
    dpo.max_num_steps=10 \
    dpo.val_batches=1 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

cd $SCRIPT_DIR
python json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

## TODO: add loss check once config is finalized
#python check_metrics.py $JSON_METRICS \
#  'data["train/loss"]["9"] < 1500' \

