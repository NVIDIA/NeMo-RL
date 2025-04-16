#!/bin/bash
set -eou pipefail

# ===== BEGIN CONFIG =====
NUM_NODES=4
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetchs metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

# Early stopping to save compute if max step has been reached
STEPS_SO_FAR=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS || echo 0)
if [[ $STEPS_SO_FAR -ge $MAX_STEPS ]]; then
    echo "[INFO] Target step $MAX_STEPS reached, skipping run"
    exit 0
fi
echo "[INFO] Steps so far: $STEPS_SO_FAR, running till $MAX_STEPS steps"

# Run the experiment
cd $PROJECT_ROOT
python -u examples/run_grpo_math.py \
    --config examples/configs/grpo_math_8B.yaml \
    policy.model_name=Qwen/Qwen2.5-7B-Instruct \
    policy.dtensor_cfg.enabled=false \
    cluster.num_nodes=$NUM_NODES \
    cluster.gpus_per_node=8 \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
python -u tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    python -u tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/token_mult_prob_error"]) < 1.1' \
        'data["train/token_mult_prob_error"]["160"] < 1.1'
fi

